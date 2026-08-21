def set_orbit_process_name(name: str) -> None:
    try:
        import setproctitle
        setproctitle.setproctitle(name)
    except (ImportError, OSError):
        pass
