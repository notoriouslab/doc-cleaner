def _run():
    try:
        from macapp.app import main
    except ImportError:
        import sys
        sys.exit(
            "錯誤：缺少 pywebview。請先安裝：pip install pywebview\n"
            "Error: pywebview is required. Install with: pip install pywebview"
        )
    main()


_run()
