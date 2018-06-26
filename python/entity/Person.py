import uuid


def create_rectangle(width, height, center_x, center_y):
    bottom_left_x = center_x - (width / 2)
    bottom_left_y = center_y - (height / 2)

    rcg_width = width + bottom_left_x
    rcg_height = height + bottom_left_y

    return bottom_left_x, bottom_left_y, rcg_width, rcg_height


class Person:
    def __init__(self, tag_name, accuracy, center_x, center_y, width, height):
        self.tag_name = tag_name
        self.accuracy = accuracy
        self.width = width
        self.height = height
        self.center_x = center_x
        self.center_y = center_y
        self.identifi_code = None

    def create_rectangle(self):
        bottom_left_x = self.center_x - (self.width / 2)
        bottom_left_y = self.center_y - (self.height / 2)
        rcg_width = self.width + bottom_left_x
        rcg_height = self.height + bottom_left_y

        return bottom_left_x, bottom_left_y, rcg_width, rcg_height
