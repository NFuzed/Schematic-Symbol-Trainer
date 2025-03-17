

class Entity(Observer):
    @typing.overload
    def __init__(self, id: str, friendly_name: str):
        super().__init__()
        self.id = id
        self.friendly_name = friendly_name
        self.images: array(cv2.typing.MatLike) = []

    def get_id(self):
        return self.id

    def get_friendly_name(self):
        return self.friendly_name

    def set_friendly_name(self, friendly_name: String):
        self.friendly_name = friendly_name

    def add_image(self, img: cv2.typing.MatLike):
        self.images.append(img)
        self.notify_observers()
        return images.len()

    def remove_image(self, img: cv2.typing.MatLike):
        self.images.remove(img)
        self.notify_observers()
