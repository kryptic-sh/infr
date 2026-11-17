//! A [`Pager`](super::Pager)'s recency order: an intrusive doubly linked list over node indices,
//! coldest first.
//!
//! A node is a slot, or a block group whose [`Pager::touch_group`](super::Pager::touch_group) is
//! in force (see `groups`). Node `0` is the sentinel: its `next` is the cold end and its `prev`
//! the hot end, so pushing at either end, moving a node and unlinking one are a few index writes
//! with no allocation. A node that is in no list links to itself.

/// The sentinel, and what [`Lru::hotter`] returns past the hot end.
pub(super) const END: u32 = 0;

#[derive(Debug, Clone, Copy)]
struct Link {
    prev: u32,
    next: u32,
}

/// What a node index stands for.
pub(super) enum Node {
    Slot(u32),
    Group(usize),
}

#[derive(Debug)]
pub(super) struct Lru {
    links: Vec<Link>,
    n_slots: u32,
}

impl Lru {
    /// An empty order over `n_slots` slot nodes, and no group nodes yet.
    pub(super) fn new(n_slots: usize) -> Self {
        let n_slots = u32::try_from(n_slots).expect("pager: n_slots must fit a u32 slot index");
        Self {
            links: (0..=n_slots).map(|n| Link { prev: n, next: n }).collect(),
            n_slots,
        }
    }

    /// The node of slot `s`.
    pub(super) fn slot(s: u32) -> u32 {
        s + 1
    }

    /// The node of group `g`. [`Self::grow_groups`] must have made room for it.
    pub(super) fn group(&self, g: usize) -> u32 {
        self.n_slots + 1 + u32::try_from(g).expect("pager: group index fits a u32")
    }

    pub(super) fn node(&self, n: u32) -> Node {
        if n <= self.n_slots {
            Node::Slot(n - 1)
        } else {
            Node::Group((n - self.n_slots - 1) as usize)
        }
    }

    /// Make room for nodes of groups `0..n_groups`.
    pub(super) fn grow_groups(&mut self, n_groups: usize) {
        let want = self.n_slots as usize + 1 + n_groups;
        while self.links.len() < want {
            let n = u32::try_from(self.links.len()).expect("pager: node index fits a u32");
            self.links.push(Link { prev: n, next: n });
        }
    }

    /// The coldest node, or [`END`] when the list is empty.
    pub(super) fn coldest(&self) -> u32 {
        self.links[END as usize].next
    }

    /// The node after `n` towards the hot end, or [`END`].
    pub(super) fn hotter(&self, n: u32) -> u32 {
        self.links[n as usize].next
    }

    pub(super) fn is_linked(&self, n: u32) -> bool {
        self.links[n as usize].next != n
    }

    pub(super) fn unlink(&mut self, n: u32) {
        let Link { prev, next } = self.links[n as usize];
        self.links[prev as usize].next = next;
        self.links[next as usize].prev = prev;
        self.links[n as usize] = Link { prev: n, next: n };
    }

    /// Link `n`, which must be in no list, just colder than `at` (or at the hot end, for [`END`]).
    pub(super) fn insert_before(&mut self, n: u32, at: u32) {
        debug_assert!(!self.is_linked(n), "pager: node {n} is linked twice");
        let prev = self.links[at as usize].prev;
        self.links[n as usize] = Link { prev, next: at };
        self.links[prev as usize].next = n;
        self.links[at as usize].prev = n;
    }

    pub(super) fn push_hot(&mut self, n: u32) {
        self.insert_before(n, END);
    }

    pub(super) fn push_cold(&mut self, n: u32) {
        self.insert_before(n, self.coldest());
    }
}
