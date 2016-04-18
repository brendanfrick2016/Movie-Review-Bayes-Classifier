# Name: 
# Date:
# Description:
#
# Change line 338 and line 23 to accurate folders containing files
from __future__ import division
import math, os, pickle, re
import time

class Bayes_Classifier:

   def __init__(self):
      """This method initializes and trains the Naive Bayes Sentiment Classifier.  If a 
      cache of a trained classifier has been stored, it loads this cache.  Otherwise, 
      the system will proceed through training.  After running this method, the classifier 
      is ready to classify input text."""
      self.good = {}
      self.bad = {}
      self.numb_good = 0
      self.numb_bad = 0 
      self.numb_reviews = 0 
      
      if os.path.isfile("/Users/ginadrutz/Desktop/Brendan/good.txt") and os.path.isfile("/Users/ginadrutz/Desktop/Brendan/bad.txt"):
         self.load("good.txt")
         self.load("bad.txt")
         try:
            self.numb_good = self.good["999"]
            self.numb_bad = self.bad["999"]
         except:
            self.numb_good = 0
            self.numb_bad = 0
         self.numb_reviews = self.numb_good + self.numb_bad
         
      else:
         self.train()


   def preprocess(self, adjust_good, adjust_bad):
      """ Reduces values of keys in good and bad based on validation testing"""
      for gword in adjust_good.keys():
         self.good[gword] = self.good[gword]*math.pow(.9,adjust_good[gword])
      for bword in adjust_bad.keys():
         self.bad[bword] = self.bad[bword]*math.pow(.9,adjust_bad[bword])

   def train(self):   
      """Trains the Naive Bayes Sentiment Classifier."""
      lFileList = []
      self.good = {}
      self.bad ={}
      y=[]
      self.average_length = [0,0]
      #x=[]
      #print good

      for fFileObj in os.walk("reviews/"):
         lFileList = fFileObj[2]
         break
      for i in lFileList:
         if "movies-1" in i:
            self.average_length[1]+=len(i)
            self.numb_bad += 1
            x=self.loadFile(i)
            y=self.tokenize(x)
            for word in y:
               if not word in self.bad:
                  self.bad['%s' %(word)]= 1
               else:
                  self.bad[word]+=1
         elif "movies-5" in i:
            self.average_length[0]+=len(i)
            self.numb_good += 1
            w=self.loadFile(i)
            j=self.tokenize(w)
            for word in j:
               if not word in self.good:
                  self.good['%s' %(word)]=1
               else:
                  self.good[word]+=1
      self.good["999"] = self.numb_good
      self.bad["999"] = self.numb_bad
      self.good["999length"] = self.average_length[0]/self.numb_good
      self.bad["999length"] = self.average_length[1]/self.numb_bad
      self.numb_reviews = self.numb_bad + self.numb_good
      
      self.save(self.good,"good.txt")                             
      self.save(self.bad,"bad.txt")

    
   def classify(self, sText):
      """Given a target string sText, this function returns the most likely document
      class to which the target string belongs (i.e., positive, negative or neutral).
      """

      class_labels = {"positive","negative"}
      class_probs = [0,0,0]
      prob_good = 0
      prob_bad = 0
      
      tokens = self.tokenize(sText)

      
      for word in tokens:
         if word in self.good.keys():
               class_probs[0] += math.log(float((self.good[word]+1)/(self.numb_good/self.numb_reviews)))
         else:
               class_probs[0] += math.log(1/(self.numb_bad/self.numb_reviews))
         if word in self.bad.keys():
               class_probs[1] += math.log(float((self.bad[word]+1)/(self.numb_bad/self.numb_reviews)))
         else:
               class_probs[1] += math.log(1/(self.numb_bad/self.numb_reviews))

 #     print "Good: ", self.numb_good/self.numb_reviews
#      print "Bad: ", self.numb_bad/self.numb_reviews

      class_probs[0] += math.log(float(self.numb_good/self.numb_reviews))
      class_probs[1] += math.log(float(self.numb_bad/self.numb_reviews))

      try:
         class_probs[0] += math.log(len(tokens)/self.average_length[0])
      except:
         class_probs[0] += 0
      try:
         class_probs[1] += math.log(len(tokens)/self.average_length[1])
      except:
         class_probs[1] += 0



      if class_probs[0] == class_probs[1]:
         return "neutral"

      elif class_probs[0] > class_probs[1]:
         return "positive"

      else:
         return "negative"

   def validate(self,lFileList = None ):#lFileList
      """feed clasify function each review and compare predicted result to true result"""
      hit = [0,0] #positive hit, negative hit
      miss = [0,0] #negative classified into positive, positive classified into negative

      adjust_good = {}
      adjust_bad = {}

      if lFileList == None:
         for fFileObj in os.walk("reviews/"):
           lFileList = fFileObj[2]
           break
      count = 0 
      for i in lFileList:
         count += 1
         if i == ".DS_Store": 
            continue
         x = self.loadFile(i)
         y = self.tokenize(x)
         temp = self.classify(y)
         if "movies-5" in i:
            result = "positive"
         elif "movies-1" in i:
            result= "negative"
         else:
            continue # other files
         if temp==result:
            if result == "positive":
               hit[0]+=1
            elif result == "negative":
               hit[1]+=1
         else: 
            if result == "negative":
               miss[0]+=1
               ## Adjust good list so classifies less
               max_pos = 0 
               for tok in y:
                  if tok in self.good:
                     if self.good[tok] > max_pos:
                        max_pos = self.good[tok]

                        if not tok in adjust_good:
                           adjust_good['%s' %(tok)]=1
                        else:
                           adjust_good[tok]+=1

            elif result == "positive":
               miss[1]+=1
               max_neg = 0

               ## Adjust bad list so classifies less
               for tok in y:
                  if tok in self.bad:
                     if self.bad[tok] > max_neg:
                        max_neg = self.bad[tok]
                        if not tok in adjust_bad:
                           adjust_bad['%s' %(tok)]=1
                        else:
                           adjust_bad[tok]+=1

         if count % (math.floor(len(lFileList)/10)) == 0:
            print "\t\t",math.ceil(count/len(lFileList)*100),"%"

      precision = [0,0]
      recall = [0,0]
      f_measure =[0,0]

      #print "Number of positive hits: ", hit[0]
      #print "Number of negative hits: ", hit[1]
      #print "Number of classified positive misses: ", miss[0]
      #print "Number of classified negative misses", miss[1]

      try:
         precision[0] = hit[0]/(hit[0]+miss[0])
      except:
         precision[0] = 0
      try:
         precision[1] = hit[1]/(hit[1]+miss[1])
      except:
         precision[1] = 0
      try:
         recall[0] = hit[0]/(hit[0]+miss[1])
      except:
         recall[0] = 0
      try:
         recall[1] = hit[1]/(hit[1]+miss[0])
      except:
         recall[1] = 0
      try:
         f_measure[0] = 2 * ((precision[0] * recall[0])/(precision[0] + recall[0]))
      except:
         f_measure[0] = 0
      try:
         f_measure[1] = 2 * ((precision[1] * recall[1])/(precision[1] + recall[1]))
      except:
         f_measure[1] = 0
      return {"precision":precision,"recall":recall,"f_measure":f_measure,"adjust_bad":adjust_bad,"adjust_good":adjust_good}


   def crossval(self):
      """goes through list of movie reviews, partitions list into 10 sections, trains on 9, tests on 1"""

      self.good = {}
      self.bad = {}
      self.numb_good = 0
      self.numb_bad = 0
      self.numb_reviews = 0

      results = {"precision": [0,0], "recall": [0,0],"f_measure":[0,0]}

      lFileList = []
      for fFileObj in os.walk("reviews/"):
         lFileList = fFileObj[2]
         break
   
      negative_lFileList = lFileList[:lFileList.index("movies-5-10.txt")]
      positive_lFileList = lFileList[lFileList.index("movies-5-10.txt"):]

      total = len(lFileList)
      num_folds = 10
      neg_subset_size = len(negative_lFileList)/num_folds
      pos_subset_size = len(positive_lFileList)/num_folds

      
      y=[]
      sum_accuracy = 0
      mean_accuracy = 0

      for i in range(num_folds):
         print "Testing Fold Number: ", i + 1
         testing_this_round = negative_lFileList[int(i*neg_subset_size):int((i+1)*neg_subset_size)] + positive_lFileList[int(i*pos_subset_size):int((i+1)*pos_subset_size)] 
         training_this_round = negative_lFileList[:int(i*neg_subset_size)]+negative_lFileList[int((i+1)*neg_subset_size):]+ positive_lFileList[:int(i*pos_subset_size)]+positive_lFileList[int((i+1)*pos_subset_size):]
         
         print "\ttraining data..."
         for i in training_this_round:
            if "movies-1" in i: 
               self.numb_bad += 1    
               x=self.loadFile(i)
               y=self.tokenize(x)
               for word in y:
                  if not word in self.bad:
                     self.bad['%s' %(word)]= 1
                  else:
                     self.bad[word]+=1
            elif "movies-5" in i:
               self.numb_good += 1
               w=self.loadFile(i)
               j=self.tokenize(w)
               for word in j:
                  if not word in self.good:
                     self.good['%s' %(word)]=1
                  else:
                     self.good[word]+=1
            self.good["999"] = self.numb_good
            self.bad["999"] = self.numb_bad
            self.numb_reviews = self.numb_bad + self.numb_good
         
         print "\tself-correcting data..."
         temp_results = self.validate(lFileList = testing_this_round)

         self.preprocess(temp_results["adjust_good"], temp_results["adjust_bad"])

         print "\tvalidating data..."
         temp_results = self.validate(lFileList = testing_this_round)
         results["precision"] = [temp_results["precision"][0] + results["precision"][0], temp_results["precision"][1] + results["precision"][1]]
         results["recall"] = [temp_results["recall"][0] + results["recall"][0], temp_results["recall"][1] + results["recall"][1]]
         results["f_measure"] = [temp_results["f_measure"][0] + results["f_measure"][0], temp_results["f_measure"][1] + results["f_measure"][1]]
      
      results["precision"] = [results["precision"][0]/num_folds,results["precision"][1]/num_folds]
      results["recall"] = [results["recall"][0]/num_folds,results["recall"][1]/num_folds]
      results["f_measure"] = [results["f_measure"][0]/num_folds,results["f_measure"][1]/num_folds]

      
      print "\nRESULTS\n"

      print "Precision:"
      print "\tGood:"
      print "\t\t",results["precision"][0]
      print "\tBad:"
      print "\t\t",results["precision"][1]

      print "Recall:"
      print "\tGood:"
      print "\t\t",results["recall"][0]
      print "\tBad:"
      print "\t\t",results["recall"][1]

      print "F_Measure:"
      print "\tGood:"
      print "\t\t",results["f_measure"][0]
      print "\tBad:"
      print "\t\t",results["f_measure"][1]





   def loadFile(self, sFilename):
      """Given a file name, return the contents of the file as a string."""

      f = open("/Users/ginadrutz/Desktop/Brendan/reviews/%s" % (sFilename), "r")
      sTxt = f.read()
      f.close()
      return sTxt
   
   def save(self, dObj, sFilename):
      """Given an object and a file name, write the object to the file using pickle."""

      f = open(sFilename, "w")
      p = pickle.Pickler(f)
      p.dump(dObj)
      f.close()
   
   def load(self, sFilename):
      """Given a file name, load and return the object stored in the file."""

      f = open(sFilename, "r")
      u = pickle.Unpickler(f)
      dObj = u.load()
      f.close()
      return dObj

   def tokenize(self, sText): 
      """Given a string of text sText, returns a list of the individual tokens that 
      occur in that string (in order)."""

      lTokens = []
      sToken = ""
      for c in sText:
         if re.match("[a-zA-Z0-9]", str(c)) != None or c == "\"" or c == "_" or c == "-":
            sToken += c
         else:
            if sToken != "":
               lTokens.append(sToken)
               sToken = ""
            if c.strip() != "":
               lTokens.append(str(c.strip()))
               
      if sToken != "":
         lTokens.append(sToken)

      return lTokens
