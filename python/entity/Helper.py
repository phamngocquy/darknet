from python.entity.Person import Person
import math


class Helper(object):

    @staticmethod
    def get_list_person(person_list_location):
        person_list = []
        for p in person_list_location:
            if p[0] == b'person':
                tag_name = p[0]
                accuracy = p[1]
                center_x = p[2][0]
                center_y = p[2][1]
                width = p[2][2]
                height = p[2][3]
                person = Person(tag_name, accuracy, center_x, center_y, width, height)
                person_list.append(person)
        return person_list

    @staticmethod
    def get_rcg_self(person_list, person):
        person_result = None
        if len(person_list) >= 1:
            min_distance = None
            for i in range(len(person_list)):
                distance = math.sqrt(math.pow(person.center_x - person_list[i].center_x, 2) + math.pow(
                    person.center_y - person_list[i].center_y, 2))
                print("tmp: ", distance)
                if (min_distance is None or min_distance > distance) and (distance < 20):
                    min_distance = distance
                    person_result = person_list[i]
            print("result: ", min_distance)
            print("//====================//")
        return person_result

    # my_person_list: current person list
    #  tmp_person_list: old person list
    @staticmethod
    def person_mapping(my_person_list, tmp_person_list):
        for i in range(len(my_person_list)):
            person = Helper.get_rcg_self(tmp_person_list, my_person_list[i])
            if person is not None and Helper.duplicate_identifi(my_person_list, person.identifi_code) is False:
                my_person_list[i].identifi_code = person.identifi_code
            else:
                my_person_list[i].identifi_code = Helper.max_identifi_code(my_person_list)

    @staticmethod
    def duplicate_identifi(my_person_list, identifi_code):
        for person in my_person_list:
            if identifi_code == person.identifi_code:
                return True
        return False

    @staticmethod
    def max_identifi_code(my_person_list):
        max_identifi = 0
        for person in my_person_list:
            tmp = person.identifi_code
            if person.identifi_code is not None and (max_identifi < tmp):
                max_identifi = tmp

        return max_identifi + 1
