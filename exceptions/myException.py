class NoPositiveError(Exception):
    def __init__(self, info):
        print(info)
