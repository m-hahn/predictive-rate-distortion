import sys

PATH = "/u/scr/corpora/ldc/2012/LDC2012T05/"

def readChineseDependencyTreebank(partition):
      if partition == "valid":
         partition = "dev"
      path = PATH+"/data/"+partition+".conll06"
      with open(path, "r") as inFile:
          data = inFile.read().strip().split("\n\n")
          if len(data) == 1:
            data = data[0].split("\r\n\r\n")
          assert len(data) > 1
      assert len(data) > 0, (language, partition, files)
      print >> sys.stderr, "Read "+str(len(data))+ " sentences from 1 "+partition+" datasets."
      return data


