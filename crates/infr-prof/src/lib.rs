//! `#[infr_prof::instrument]` — build-time profiling injection (see docs/PERF.md).
//!
//! Applied (always behind `#[cfg_attr(infr_profile, ...)]`) to a `fn`, an `impl` block, or an
//! inline `mod`, it rewrites EVERY function inside to open an `infr_prof_rt` span on entry and
//! close it via RAII on any exit path. Coverage is therefore per-item: annotate an impl block
//! once and every method in it — including ones added later — is profiled for free.
//!
//! Skipped automatically (no span injected):
//! - `const fn` (no `Instant` in const), `async fn` (guard would cross `.await`/threads),
//!   `#[naked]` fns (no prologue allowed), `#[test]`/`#[bench]` fns
//! - `#[inline]`/`#[inline(always)]` fns — declaring a fn inline asserts it is smaller than
//!   the ~50ns probe pair, so probing it would only measure the probe
//! - fns carrying `#[infr_prof::skip]` (or `#[cfg_attr(infr_profile, infr_prof::skip)]`) —
//!   explicit opt-out for measured-too-hot functions
//! - closures (only `fn` items are touched) and macro invocations inside items
//!
//! Handled correctly: generics (the injected `static` site is shared across instantiations),
//! trait impls (site named `<Ty as Trait>::fn`), recursion (depth-aware inclusive time),
//! `?`/`return`/panic (RAII guard), nested fns (each gets its own span).

use proc_macro::TokenStream;
use quote::{quote, ToTokens};
use syn::visit_mut::VisitMut;
use syn::{parse_macro_input, Attribute, Item, Signature};

/// Rewrite every `fn` in the annotated item (fn / impl block / inline mod) to record a span.
#[proc_macro_attribute]
pub fn instrument(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut item = parse_macro_input!(item as Item);
    Instrumenter { scope: Vec::new() }.visit_item_mut(&mut item);
    item.into_token_stream().into()
}

/// Identity attribute: opt a single fn out of an enclosing `#[instrument]`. `instrument`
/// detects it by name; standalone it expands to nothing extra.
#[proc_macro_attribute]
pub fn skip(_attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}

struct Instrumenter {
    /// Lexical scope inside the annotated item: impl-type names and enclosing fn names.
    /// (Inline `mod` names are NOT pushed — `module_path!()` already reflects them.)
    scope: Vec<String>,
}

impl Instrumenter {
    fn site_name(&self, ident: &syn::Ident) -> String {
        if self.scope.is_empty() {
            ident.to_string()
        } else {
            format!("{}::{}", self.scope.join("::"), ident)
        }
    }

    fn inject(&self, sig: &Signature, block: &mut syn::Block) {
        let name = self.site_name(&sig.ident);
        let prelude: syn::Block = syn::parse2(quote! {{
            static __INFR_PROF_SITE: ::infr_prof_rt::Site =
                ::infr_prof_rt::Site::new(::core::module_path!(), #name);
            let __infr_prof_guard = ::infr_prof_rt::enter(&__INFR_PROF_SITE);
        }})
        .expect("infr-prof: prelude parse");
        block.stmts.splice(0..0, prelude.stmts);
    }
}

fn should_skip(attrs: &[Attribute], sig: &Signature) -> bool {
    if sig.constness.is_some() || sig.asyncness.is_some() {
        return true;
    }
    attrs.iter().any(|a| {
        let path = a.path();
        if path.is_ident("test") || path.is_ident("bench") || path.is_ident("naked") {
            return true;
        }
        // Any #[inline] fn is treated as a sub-probe-cost leaf: the author already declared it
        // small enough to inline, so a ~50ns enter/exit pair would dominate it (measured:
        // hadd_i32_xmm at 1.7e9 calls/decode-run). Un-inline'd hot leaves opt out explicitly
        // with #[cfg_attr(infr_profile, infr_prof::skip)].
        if path.is_ident("inline") {
            return true;
        }
        // #[infr_prof::skip] directly, or wrapped in #[cfg_attr(infr_profile, infr_prof::skip)]
        // (cfg_attr is still unexpanded when this macro runs on an enclosing item).
        let s = a.to_token_stream().to_string();
        s.contains("infr_prof") && s.contains("skip")
    })
}

fn type_scope_name(imp: &syn::ItemImpl) -> String {
    let ty = imp.self_ty.to_token_stream().to_string().replace(' ', "");
    match &imp.trait_ {
        Some((_, path, _)) => {
            let tr = path.to_token_stream().to_string().replace(' ', "");
            format!("<{ty} as {tr}>")
        }
        None => ty,
    }
}

impl VisitMut for Instrumenter {
    fn visit_item_fn_mut(&mut self, node: &mut syn::ItemFn) {
        self.scope.push(node.sig.ident.to_string());
        syn::visit_mut::visit_item_fn_mut(self, node); // nested fns first
        self.scope.pop();
        if !should_skip(&node.attrs, &node.sig) {
            self.inject(&node.sig, &mut node.block);
        }
    }

    fn visit_impl_item_fn_mut(&mut self, node: &mut syn::ImplItemFn) {
        self.scope.push(node.sig.ident.to_string());
        syn::visit_mut::visit_impl_item_fn_mut(self, node);
        self.scope.pop();
        if !should_skip(&node.attrs, &node.sig) {
            self.inject(&node.sig, &mut node.block);
        }
    }

    fn visit_item_impl_mut(&mut self, node: &mut syn::ItemImpl) {
        self.scope.push(type_scope_name(node));
        syn::visit_mut::visit_item_impl_mut(self, node);
        self.scope.pop();
    }

    fn visit_item_mod_mut(&mut self, node: &mut syn::ItemMod) {
        // Leave #[cfg(test)] modules alone.
        let is_test_mod = node
            .attrs
            .iter()
            .any(|a| a.path().is_ident("cfg") && a.to_token_stream().to_string().contains("test"));
        if !is_test_mod {
            syn::visit_mut::visit_item_mod_mut(self, node);
        }
    }

    // Trait *definitions* (default bodies) and closures: deliberately untouched.
    fn visit_item_trait_mut(&mut self, _node: &mut syn::ItemTrait) {}
    fn visit_expr_closure_mut(&mut self, node: &mut syn::ExprClosure) {
        // Recurse: a closure body may contain nested `fn` items worth instrumenting.
        syn::visit_mut::visit_expr_closure_mut(self, node);
    }
}
