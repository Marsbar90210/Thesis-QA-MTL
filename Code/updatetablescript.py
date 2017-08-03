import sys
from pprint import pprint
from functools import reduce
from datetime import datetime

def getdata(data):
    ress = []
    cqa = None
    diter = iter(data)
    for line in diter:
        if (line == ""):
            break
        line = line.strip()
        fword = line.split(" ")[0].lower()
        if(fword == "output"):
            # New output started
            ress.append({})
        if(fword == "running"):
            # Extract language and date
            parts = line.split(",")
            ress[-1]["ver"] = ver = parts[0].split(" ")[-1]
            ress[-1]["date"] = date = parts[-1].strip()
        if(fword == "parameters"):
            # Parameters line
            parts = line.replace(" ", "").split(",")[1:]
            param = [(temp[0], temp[1]) for temp in map(lambda x:x.split("="), parts)]
            ress[-1]["param"] = param
        if(fword == "training"):
            # New training line
            qa = line.split(" ")[1]
            line = next(diter)
            while line.split(" ")[0].strip().lower() != "training":
                line = next(diter)
            time = line.split(" ")[2]
            ress[-1][qa] = ress[-1].get(qa, [])
            ress[-1][qa].append({"time":time})
            cqa = qa
        if(fword == "train"):
            # Train line
            parts = line.split(":")
            key = " ".join(parts[0].split(" ")[1:])
            tdata = [(temp[0], temp[1]) for temp in
                map(lambda x:x.split("="), parts[-1].strip().split(" "))]
            ress[-1][cqa][-1][key] = tdata
        if(fword == "test"):
            # Total line
            parts = line.replace(" / ", "/").split("=")
            parts[0] = " ".join(parts[0].split(" ")[0:3])
            for temp in zip(*map(lambda x:x.lower().strip().split("/"), parts)):
                ress[-1][cqa][-1][temp[0]] = ress[-1][cqa][-1].get(temp[0], [])
                ress[-1][cqa][-1][temp[0]].append(temp[1])
    return ress

def reducer(x, y):
    if (datetime.strptime(x["date"], "%d/%m/%Y") > datetime.strptime(y["date"], "%d/%m/%Y")):
        return x
    else:
        return y

def reduceiter(data, idx=None):
    best = data[0]
    data = data[1:]
    key = "acc";
    if (idx != None):
        key += " {}".format(idx);
    for temp in data:
        i1 = float(dict(best[key])["final"])
        i2 = float(dict(temp[key])["final"])
        if (i1 < i2):
            best = temp
    return best

def reduceiter_avg(data, idx=None):
    modifier = len(data)
    accumulator = 0
    key = "test accuracy";
    if (idx):
        key += " {}".format(idx);
    for temp in data:
        accumulator += float(temp[key][idx])
    data[0]["test accuracy"] = [accumulator/modifier]
    return data[0]

def gettestacc(elem, qelem, ver):
    if (len(qelem["test accuracy"]) == 1):
        return qelem["test accuracy"][0]
    else:
        return qelem["test accuracy"][elem["ver"].split("/").index(ver)]

model = sys.argv[1]
ver = "en"
reverseIndex = 0
delete = 0
if (len(sys.argv) > 2):
    ver = sys.argv[2]
if (len(sys.argv) > 3):
    reverseIndex = int(sys.argv[3])
if (len(sys.argv) > 4):
    delete = int(sys.argv[4])
data = [];
pdata = [];
idx = 0;
if (not delete):
    with open("../outputs/output{}.txt".format(model), "r") as inputfile:
        odata = inputfile.readlines()
        pdata = getdata(odata)
        pdata = filter(lambda x:ver in x["ver"], pdata)
with open("../acl short paper/data.tex", "r") as tablefile:
    data = tablefile.read().splitlines()
    if (len(data) == 0):
        data.append("\\begin{tabular}{*{3}{P{1cm}}}")
        data.append("task/method&E2E baseline&DMN+\\\\")
        for i in range(20):
            data.append("{}&&\\\\".format(i+1))
        data.append("\\end{tabular}")
    tag = model + " " + ver
    columns = data[2].replace("\\\\", "").split("&")
    if (tag in columns):
        idx = columns.index(tag)
    else:
        idx = -1
if (not delete):
    pdata = sorted(pdata, key=lambda k: k["date"], reverse=True)
    elem = pdata[-1 - reverseIndex];
    if (idx < 0):
        data[0] = "\\begin{{tabular}}{{*{{{}}}{{P{{1cm}}}}}}".format(len(data[1].split("&")) + 1)
        data[1] = data[1].replace("\\\\", "&") + model + " " + ver + "\\\\"
        data[2] = data[2].replace("\\\\", "&") + model + " " + ver + "\\\\"
        total_acc = 0.0
        for i in range(20):
            qelem = reduceiter_avg(elem["qa{}".format(i+1)], 0)
            num = float(gettestacc(elem, qelem, ver))*100
            total_acc += num
            data[i+3] = data[i+3].replace("\\\\", "&") + "{:.1f}".format(num) + "\\\\"
        avg = total_acc/20
        data[-2] = data[-2].replace("\\\\", "&") + "{:.1f}".format(avg)
    else:
        total_acc = 0.0
        for i in range(20):
            qelem = reduceiter_avg(elem["qa{}".format(i+1)], 0)
            num = float(gettestacc(elem, qelem, ver))*100
            total_acc += num
            temp = data[i+3].replace("\\\\", "").split("&")
            data[i+3] = "{}{}{}".format("&".join(temp[0:idx]),
                                        "&{:.1f}&".format(num),
                                        "&".join(temp[idx+1:]))
            if(data[i+3][-1] == "&"):
                data[i+3] = data[i+3][:-1]
            data[i+3] = data[i+3] + "\\\\"
        avg = total_acc/20
        temp = data[-2].replace("\\\\", "").split("&")
        data[-2] = "{}{}{}".format("&".join(temp[0:idx]),
                                   "&{:.1f}&".format(avg),
                                   "&".join(temp[idx+1:]))
        if(data[-2][-1] == "&"):
            data[-2] = data[-2][:-1]
        data[-2] = data[-2]
else:
    if (idx > -1):
       data[0] = "\\begin{{tabular}}{{*{{{}}}{{P{{1cm}}}}}}".format(len(data[1].split("&")) - 1)
       data[1] = data[1].replace("&" + model + " " + ver, "")
       data[2] = data[2].replace("&" + model + " " + ver, "")
       for i in range(20):
            temp = data[i+3].replace("\\\\", "").split("&")
            data[i+3] = "{}{}{}".format("&".join(temp[0:idx]),
                                        "&",
                                        "&".join(temp[idx+1:]))
            if(data[i+3][-1] == "&"):
                data[i+3] = data[i+3][:-1]
            data[i+3] = data[i+3] + "\\\\"
       temp = data[-2].replace("\\\\", "").split("&")
       data[-2] = "{}{}{}".format("&".join(temp[0:idx]),
                                  "&",
                                  "&".join(temp[idx+1:]))
       if(data[-2][-1] == "&"):
           data[-2] = data[-2][:-1]
       data[-2] = data[-2]
with open("../acl short paper/data.tex", "w") as file:
    file.write("\n".join(data))
