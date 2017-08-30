import json, sys, os
from watson_developer_cloud import SpeechToTextV1


#print(json.dumps(speech_to_text.models(), indent=2))

#print(json.dumps(speech_to_text.get_model('en-US_BroadbandModel'), indent=2))

def processSoundFile( speech_to_text,soundFile ):
  with open(soundFile,'rb') as audio_file:
      s2tout=speech_to_text.recognize(
             audio_file, content_type='audio/wav', timestamps=False,
             model='en-US_NarrowbandModel',speaker_labels=True, max_alternatives=1,
             word_confidence=True,smart_formatting=True)
  return s2tout

def parseFromToUtterance( s2tout ):

#
#  s2tout is a dictionary with four keys: speaker_labels, results, result_index, warnings   
#       each key is a list of dictionaries, primarily:
#           speaker_labels: contains To and From timestamps of every spoken word by a speaker  
#           results: contains the whole utterance by a speaker with To and From timestamps for each spoken word  
#                    we will parse the To and From timestamps into utteranceStart and utteranceEnd  
#  

   for i in range(s2tout['results'].__len__()):
      utteranceStart=s2tout['results'][i]['alternatives'][0]['timestamps'][0][1]
      numberWords=s2tout['results'][i]['alternatives'][0]['timestamps'].__len__()
      utteranceEnd=s2tout['results'][i]['alternatives'][0]['timestamps'][numberWords-1][2]
      utterance=s2tout['results'][i]['alternatives'][0]['transcript']
      print("%s,%s,\"%s\"" % (utteranceStart,utteranceEnd,utterance))
   return

def parseFromToSpeakerLabels ( s2tout ):

   wrk_speakerStart=''
   wrk_speakerEnd=''
   wrk_speakerID=''

   for i in range(s2tout['speaker_labels'].__len__()):
      if wrk_speakerID != s2tout['speaker_labels'][i]['speaker']:
         if wrk_speakerID != '':
           print("%s,%s,%s" % (wrk_speakerStart,wrk_speakerEnd,wrk_speakerID))
           wrk_speakerID    = s2tout['speaker_labels'][i]['speaker']   
           wrk_speakerStart = s2tout['speaker_labels'][i]['from']   
           wrk_speakerEnd   = s2tout['speaker_labels'][i]['to']   
         else:
           wrk_speakerID    = s2tout['speaker_labels'][i]['speaker']   
           wrk_speakerStart = s2tout['speaker_labels'][i]['from']   
           wrk_speakerEnd   = s2tout['speaker_labels'][i]['to']   
      else:
         wrk_speakerEnd=s2tout['speaker_labels'][i]['to']   
   return

def main( speech_to_text, soundFile ):
    s2tout=processSoundFile(speech_to_text,soundFile)
    print("Parsing results ...")
    parseFromToUtterance(s2tout)
    print("Parsing speaker_labels ...")
    parseFromToSpeakerLabels(s2tout)


if __name__ == '__main__':

#
# Make sure sound file is passed to us, relative path+name or asolute path+name 
#

  if len(sys.argv) != 2:
     sys.exit("Error: Please provide the name of the sound file as the only input parm")
  else:
     soundFile = sys.argv[1]
  if (os.environ["S2T_USERNAME"] == '' or os.environ["S2T_PASSWORD"] == ''):
     sys.exit("Error: Be sure to set system variables S2T_USERNAME and S2T_PASSWORD to appropriate values")
 
#
#  Log into S2T
#
  speech_to_text = SpeechToTextV1(
    username=os.environ["S2T_USERNAME"],
    password=os.environ["S2T_PASSWORD"],
    x_watson_learning_opt_out=False
  )

# Invoke main()

  main(speech_to_text, soundFile)
