#!/usr/bin/env python
# -*- coding: utf-8 -*-

PATH = "/u/scr/mhahn/CORPORA/LDC95T8/"

LDC95T8 = []
wordCount = 0.0
print("Reading LDC95T8")
with open(PATH+"/nikkei2-tagged.txt", "r") as inFile:
    for line in inFile:
       line = line.strip()
       line= line.split(" ")
       if len(line) < 4:
          continue
       if line[1].startswith("DOCNO"):
            continue
       if line[1].startswith("DATE"):
            continue
       if line[1].startswith("SUMMARY"):
            continue

       if line[0].startswith("<") and line[2].startswith(">"):
          line = line[4:]
       if line[0].startswith("<") and line[3].startswith(">"):
          line = line[4:]
#       if len(line) > 5 and line[-4].startswith("<") and line[-1].startswith(">"):
#           line = line[:-4]
#       if len(line) > 4 and line[-3].startswith("<") and line[-1].startswith(">"):
#           line = line[:-3]
       if len(line) < 2:
           continue
       line = [x.split("/") for x in line]
       line = [x for x in line if x[1] != "補助記号" and x[1] != "空白"]
       line = [{"word" : x[0], "posUni" : x[1]} for x in line]
#       print(str(len(line))+" "+(" ".join(["/".join(x) for x in line])))
#       print(line)
       LDC95T8.append(line)
       wordCount += len(line)
       if len(LDC95T8) % 5000 == 0:
          print(wordCount/2e6)
       if wordCount > 2e6:
          break

import random

random.Random(5).shuffle(LDC95T8)
print(len(LDC95T8))


