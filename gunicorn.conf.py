# Render may preload the app, so the recording threads start here, inside the worker that also serves /status.
def post_fork(server, worker):
    import app
    app.start()
