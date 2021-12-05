import os

path = "./log"
path2 = "./parsed_log/"

file_list = os.listdir(path)
batch = 1563

for file_name in file_list:
    new_path = path + "/" + file_name
    with open(new_path) as f:
        text = f.readlines()

        idx = 1
        result = 0
        store = []

        for txt in text:
            if idx == batch:
                idx = 1
                store.append(result / batch)
                result = 0
            if "torch" in txt:
                continue
            num = float(txt.replace('\n', ''))
            result += num
            idx += 1

        with open(path2+file_name, 'a') as f2:
            for content in store:
                f2.write(str(content) + "\n")
