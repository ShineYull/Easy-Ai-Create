

class UIHandler:

    def text_handler(self, name):
        return "Hello " + name + "!"

    def audio_handler(self):
        pass

    def auth_handler(self, username, password):
        return username == password