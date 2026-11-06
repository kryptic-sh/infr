//! Inherited systemd TCP listeners.

use std::{io, net::TcpListener};

/// Duplicate systemd's sole inherited TCP listener on Linux; elsewhere return `None`.
pub fn activated_tcp_listener() -> io::Result<Option<TcpListener>> {
    platform::activated_tcp_listener()
}

#[cfg(not(target_os = "linux"))]
mod platform {
    use super::*;

    pub(super) fn activated_tcp_listener() -> io::Result<Option<TcpListener>> {
        Ok(None)
    }
}

#[cfg(target_os = "linux")]
mod platform {
    use super::*;
    use std::os::{
        fd::{AsRawFd, FromRawFd, OwnedFd, RawFd},
        unix::fs::MetadataExt,
    };

    fn invalid(message: &str) -> io::Error {
        io::Error::new(io::ErrorKind::InvalidInput, message)
    }

    fn activation_count(pid: Option<&str>, fds: Option<&str>, current: u32) -> io::Result<bool> {
        let Some(pid) = pid else {
            return Ok(false);
        };
        let pid = pid
            .parse::<u32>()
            .map_err(|_| invalid("invalid systemd LISTEN_PID"))?;
        if pid != current {
            return Ok(false);
        }
        let fds = fds
            .ok_or_else(|| invalid("systemd LISTEN_FDS is missing"))?
            .parse::<u32>()
            .map_err(|_| invalid("invalid systemd LISTEN_FDS"))?;
        match fds {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(invalid("infr requires exactly one systemd TCP listener")),
        }
    }

    fn socket_option(fd: RawFd, option: libc::c_int) -> io::Result<libc::c_int> {
        let mut value: libc::c_int = 0;
        let mut len = std::mem::size_of_val(&value) as libc::socklen_t;
        // Both output pointers refer to initialized writable values of the advertised sizes.
        let result = unsafe {
            libc::getsockopt(
                fd,
                libc::SOL_SOCKET,
                option,
                (&mut value as *mut libc::c_int).cast(),
                &mut len,
            )
        };
        if result < 0 {
            return Err(io::Error::last_os_error());
        }
        if len as usize != std::mem::size_of_val(&value) {
            return Err(invalid("unexpected systemd socket option size"));
        }
        Ok(value)
    }

    fn duplicate_listener(fd: RawFd) -> io::Result<TcpListener> {
        // F_DUPFD_CLOEXEC creates a new descriptor without taking ownership of the supplied fd.
        let duplicate = unsafe { libc::fcntl(fd, libc::F_DUPFD_CLOEXEC, 3) };
        if duplicate < 0 {
            return Err(io::Error::last_os_error());
        }
        // The successful duplication returned a fresh descriptor owned exclusively by this call.
        let owned = unsafe { OwnedFd::from_raw_fd(duplicate) };
        let fd = owned.as_raw_fd();
        if !matches!(
            socket_option(fd, libc::SO_DOMAIN)?,
            libc::AF_INET | libc::AF_INET6
        ) || socket_option(fd, libc::SO_TYPE)? != libc::SOCK_STREAM
            || socket_option(fd, libc::SO_PROTOCOL)? != libc::IPPROTO_TCP
            || socket_option(fd, libc::SO_ACCEPTCONN)? == 0
        {
            return Err(invalid("systemd fd 3 is not a listening TCP socket"));
        }
        let listener = TcpListener::from(owned);
        listener.set_nonblocking(true)?;
        Ok(listener)
    }

    fn check_pidfd_identity(value: &str) -> io::Result<()> {
        let expected = value
            .parse::<u64>()
            .map_err(|_| invalid("invalid systemd LISTEN_PIDFDID"))?;
        // pidfd_open returns a new descriptor for this process without borrowing Rust-owned fds.
        let fd = unsafe { libc::syscall(libc::SYS_pidfd_open, std::process::id(), 0) };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        // The successful syscall returned a fresh descriptor owned exclusively by this call.
        let fd = unsafe { OwnedFd::from_raw_fd(fd as RawFd) };
        if std::fs::File::from(fd).metadata()?.ino() != expected {
            return Err(invalid(
                "systemd LISTEN_PIDFDID does not identify this process",
            ));
        }
        Ok(())
    }

    fn environment(name: &str) -> io::Result<Option<String>> {
        match std::env::var(name) {
            Ok(value) => Ok(Some(value)),
            Err(std::env::VarError::NotPresent) => Ok(None),
            Err(std::env::VarError::NotUnicode(_)) => {
                Err(invalid(&format!("invalid systemd {name}")))
            }
        }
    }

    pub(super) fn activated_tcp_listener() -> io::Result<Option<TcpListener>> {
        let pid = environment("LISTEN_PID")?;
        let fds = environment("LISTEN_FDS")?;
        if !activation_count(pid.as_deref(), fds.as_deref(), std::process::id())? {
            return Ok(None);
        }
        if let Some(identity) = environment("LISTEN_PIDFDID")? {
            check_pidfd_identity(&identity)?;
        }
        // F_GETFD only reads descriptor flags and does not transfer ownership.
        let flags = unsafe { libc::fcntl(3, libc::F_GETFD) };
        if flags < 0 {
            return Err(io::Error::last_os_error());
        }
        // Preserve the inherited descriptor's flags and prevent leaking it across exec.
        if unsafe { libc::fcntl(3, libc::F_SETFD, flags | libc::FD_CLOEXEC) } < 0 {
            return Err(io::Error::last_os_error());
        }
        duplicate_listener(3).map(Some)
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::net::{TcpStream, UdpSocket};
        use std::os::unix::net::UnixListener;

        #[test]
        fn metadata_requires_one_fd_for_this_process() {
            for (pid, fds, expected) in [
                (None, None, false),
                (None, Some("1"), false),
                (Some("99"), Some("bad"), false),
                (Some("42"), Some("0"), false),
                (Some("42"), Some("1"), true),
            ] {
                assert_eq!(activation_count(pid, fds, 42).unwrap(), expected);
            }
            for (pid, fds) in [
                (Some("bad"), Some("1")),
                (Some("42"), None),
                (Some("42"), Some("bad")),
                (Some("42"), Some("2")),
                (Some("42"), Some("-1")),
            ] {
                assert!(activation_count(pid, fds, 42).is_err());
            }
        }

        #[test]
        fn duplicates_listener_with_nonblocking_and_cloexec() {
            let original = TcpListener::bind("127.0.0.1:0").unwrap();
            let listener = duplicate_listener(original.as_raw_fd()).unwrap();
            assert_eq!(
                listener.local_addr().unwrap(),
                original.local_addr().unwrap()
            );
            assert_ne!(listener.as_raw_fd(), original.as_raw_fd());
            // Both fcntl calls inspect open descriptors without changing ownership.
            unsafe {
                assert_ne!(
                    libc::fcntl(listener.as_raw_fd(), libc::F_GETFD) & libc::FD_CLOEXEC,
                    0
                );
                assert_ne!(
                    libc::fcntl(listener.as_raw_fd(), libc::F_GETFL) & libc::O_NONBLOCK,
                    0
                );
            }
            drop(listener);
            assert!(original.local_addr().is_ok());
        }

        #[test]
        fn accepts_ipv6_when_available() {
            let Ok(original) = TcpListener::bind("[::1]:0") else {
                return;
            };
            assert_eq!(
                duplicate_listener(original.as_raw_fd())
                    .unwrap()
                    .local_addr()
                    .unwrap(),
                original.local_addr().unwrap()
            );
        }

        #[test]
        fn rejects_nonlisteners_without_closing_the_original() {
            let file = tempfile::tempfile().unwrap();
            let udp = UdpSocket::bind("127.0.0.1:0").unwrap();
            let dir = tempfile::tempdir().unwrap();
            let unix = UnixListener::bind(dir.path().join("socket")).unwrap();
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let stream = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
            for fd in [
                file.as_raw_fd(),
                udp.as_raw_fd(),
                unix.as_raw_fd(),
                stream.as_raw_fd(),
                -1,
            ] {
                assert!(duplicate_listener(fd).is_err());
            }
            assert!(file.metadata().is_ok());
            assert!(udp.local_addr().is_ok());
            assert!(unix.local_addr().is_ok());
            assert!(stream.peer_addr().is_ok());
        }

        #[test]
        fn invalid_pidfd_identity_is_rejected() {
            assert!(check_pidfd_identity("bad").is_err());
            assert!(check_pidfd_identity("0").is_err());
        }
    }
}
