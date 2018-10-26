import os
import sys
import shutil


"""
RecordingData 
          /sw_45001               <- the recording dir
               /0001.6300         <- 1st superframe binary file.  Here speaker 6300 is speaking.
               /0002.6300         <- 2nd superframe and 6300 is still speaking.
   
"""

# Parent directory containing all the recording.  There is one recording per subdirectory.
all_files_path = 'C:\Diarization\RecordingDataAll'

# Get a list of all the recording directory names
dirs = [name for name in os.listdir(all_files_path)]
allRecordingDirNames = [name for name in dirs if name.startswith('sw_')]

# Get list of training recording dirs
training_recording_path = 'C:\Diarization\RNNTrainingData-Trim'
dirs = [name for name in os.listdir(training_recording_path)]
trainingRecordingDirNames = [name for name in dirs if name.startswith('sw_')]

# Create a list of testRecordingDirNames = allRecordingDirNames - trainingRecordingDirNames
testRecordingDirNames = [name for name in allRecordingDirNames if name not in trainingRecordingDirNames]

print("allRecordingDirNames all recordings count                : %d" % len(allRecordingDirNames))
print("trainingRecordingDirNames recording for R-CNN training   : %d" % len(trainingRecordingDirNames))
print("testRecordingDirNames all recs - training recs = test    : %d" % len(testRecordingDirNames))

testFullPaths = [os.path.join(all_files_path, fullPath) for fullPath in testRecordingDirNames]
print(testFullPaths)

testRecordingsDir = 'C:\Diarization\RecordingDataTest'
with open("CopyTestRecordings.bat", 'w') as outFile:
    for path in testFullPaths:
        subdir = os.path.basename(path)
        copyDir = testRecordingsDir + '\\' + subdir

        print("mkdir %s" % copyDir)
        print("copy %s  %s" % (path, copyDir))
        outFile.write("mkdir %s \n" % copyDir)
        outFile.write("copy %s  %s \n" % (path, copyDir))




