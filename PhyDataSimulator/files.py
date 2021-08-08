import os
import pickle
import shutil


# sets = ['_path', '_sets', 'type', 'set_type', 'delete', 'get', 'list', 'save', 'path']


def nor(data):
    txt = ""
    srt1 = ord('0')
    end1 = ord('9')
    srt2 = ord("A")
    end2 = ord("Z")
    srt3 = ord("a")
    end3 = ord("z")
    delta = end3 - srt1
    data = list(data)
    for i in data:
        get = ord(i)
        while True:
            get = srt1 + get % delta
            if srt1 <= get <= end1 or srt2 <= get <= end2 or srt3 <= get <= end3:
                break
            else:
                get *= 2
        txt += chr(get)
    return txt


def lock(data, step=1):
    data = list(data)
    if step > 12.0:
        showerror("Error:", "step must less than 12.0")
        return
    j = 0
    for i in range(len(data)):
        if j >= 64:
            j = 0
        else:
            j += 1
        data[i] = chr(ord(data[i]) + 256 + int(j * step))
    txt = ""
    for i in data:
        txt += i
    return txt


def unlock(data, step=1):
    data = list(data)
    if step > 12.0:
        showerror("Error:", "step must less than 12.0")
        return
    j = 0
    for i in range(len(data)):
        if j >= 64:
            j = 0
        else:
            j += 1
        data[i] = chr(ord(data[i]) - 256 - int(j * step))
    txt = ""
    for i in data:
        txt += i
    return txt


def x16encode(data):
    if isinstance(data, str):
        head = 1
    elif isinstance(data, int):
        head = 2
    elif isinstance(data, float):
        head = 3
    else:
        showerror("Error:", "UnSupport Type '" + type(data) + "'")
        return
    data = str(head) + str(data)
    x16_data = data.encode().hex()
    lock_data = lock(x16_data)
    return lock_data


def x16decode(hex_data):
    try:
        if isinstance(hex_data, int):
            showerror("Error:", "data must be x16 str, not int")
            return
        hex_data = unlock(hex_data)
        str_data = bytearray.fromhex(hex_data).decode()
        if str_data[:1] == '1':
            return str_data[1:]
        elif str_data[:1] == '2':
            return int(str_data[1:])
        elif str_data[:1] == '3':
            return float(str_data[1:])
    except:
        return None
def Get(n):
	a = f.get(n)['answer']
	t = []
	for k in a:
		t += [x16decode(a[k])]
	return t


class files:
    """
    eg.:
    d = files()
    d.get("xxx") (等效于d["xxx"] a.xxx)
    d.get("123") (等效于d["xxx"])
    d.save("123",XXOO)
    d.delete("123")
    """

    def __init__(self, root_path=""):
        """
        数据文件管理器，请在创建时指定数据记录位置，默认当前目录
        :param root_path: 数据储存位置(str)
        """
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        self.type = '.data'
        self._path = root_path
        self._sets = self.list(False)

    def set_type(self, file_type='.data'):
        self.type = file_type

    def delete(self, k, update=True):
        """
        删除一项数据
        :param k: 识别符(object)，建议传入带有__str__方法的对象
        :param update: 是否更新数据表单(default=False),更新后使用get的快速功能将不会不准确, 如果不使用快速功能建议为False
        :return: 成功返回True，失败返回False
        """
        temp = [i if i[len(i) - 5:] == self.type else "" for i in os.listdir(self._path if self._path else None)]
        if k is ...:
            self.__del_file__(self._path)
            self._sets = []
            return
        k = str(k)
        try:
            os.remove(os.path.join(self._path, k + self.type))
            if update:
                temp.remove(k + self.type)
                self._sets = temp
            return True
        except:
            return False

    def get(self, k, quick=True):
        """
        获取一项被保存的数据
        :param k: 识别符(object)，建议传入带有__str__方法的对象
        :param quick: 是否为快速模式，默认True
        :return: 成功返回数据，失败返回None
        """
        if k is ...:
            return [self.get(i) for i in self.list()]
        k = str(k)
        if quick:
            if k in self._sets:
                return pickle.load(open(os.path.join(self._path, k + self.type), "rb"))
        else:
            if k in self.list():
                return pickle.load(open(os.path.join(self._path, k + self.type), "rb"))
        return None

    def list(self, update=True):
        """
        列举所有数据项的文件名
        :param update: 是否更新数据表单(default=True),更新后使用get的快速功能将不会不准确
        :return: list[str，str...]
        """
        temp = [i[:-5] if i[-5:] == self.type else "" for i in os.listdir(self._path if self._path else None)]
        temp = list(set(temp))
        if '' in temp:
            temp.remove('')
        if update:
            self._sets = temp
        return temp

    def save(self, k, content, update=True):
        """
        保存一段数据
        :param k: 识别符(object)，建议传入带有__str__方法的对象
        :param content: 数据内容(object), 如果更换了电脑，请确保对应的module存在, or更换后的电脑读不出来
        :param update: 是否更新数据表单(default=True)
        """
        pickle.dump(content, open(os.path.join(self._path, k + self.type), "wb"))
        if update:
            self._sets.append(k)

    def path(self):
        """
        获得绑定路径
        :return: str
        """
        return self._path

    def __del_file__(self, filepath):
        """
        删除某一目录下的所有文件或文件夹
        :param filepath: 路径
        :return:
        """
        del_list = os.listdir(filepath)
        for f in del_list:
            file_path = os.path.join(filepath, f)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

    def __getattr__(self, attr):
        return self.get(attr)

    def __getitem__(self, item):
        return self.get(item)
