import aext_assistant_server

from ._version import __version__


def _jupyter_labextension_paths():
    return [{"src": "labextension", "dest": "@anaconda/assistant"}]


def _jupyter_server_extension_points():
    return [{"module": "aext_assistant"}]


def _load_jupyter_server_extension(server_app):
    web_app = server_app.web_app
    base_url = web_app.settings["base_url"]
    routes = aext_assistant_server.get_routes(base_url)
    web_app.add_handlers(".*$", routes)
    server_app.log.info("Registered aext_assistant server extension")


load_jupyter_server_extension = _load_jupyter_server_extension
