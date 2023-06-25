import pymysql

class DBManage:

    def __init__(self):
        self.conn = pymysql.connect()