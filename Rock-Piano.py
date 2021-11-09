#!/usr/bin/env python
# coding: utf-8

# # Rock Piano (ver. 1.0)
# 
# ## "Music comes from the heart!" ---LU
#  
# ***
#  
# ### Powered by tegridy-tools TMIDIX Optimus Processors: https://github.com/Tegridy-Code/tegridy-tools
#  
# ***
#  
# ### Credit for GPT2-RGA code used in this colab goes out @ Sashmark97 https://github.com/Sashmark97/midigen and @ Damon Gwinn https://github.com/gwinndr/MusicTransformer-Pytorch
#  
# ***
#  
# ### WARNING: This complete implementation is a functioning model of the Artificial Intelligence. Please excercise great humility, care, and respect. https://www.nscai.gov/
#  
# ***
#  
# #### Project Los Angeles
#  
# #### Tegridy Code 2021
#  
# ***

# In[ ]:


get_ipython().system('nvidia-smi')


# In[1]:


#@title Install all dependencies (run only once per session)
    
get_ipython().system('git clone https://github.com/asigalov61/Rock-Piano    ')
    
get_ipython().system('pip install tqdm')


# In[3]:


print('Loading needed modules. Please wait...')
import os
from datetime import datetime
import secrets
import copy
import tqdm
from tqdm import tqdm

os.chdir('Rock-Piano')

print('Loading TMIDIX module...')
import TMIDIX
print('Loading GPT2RGA module...')
from GPT2RGA import *


# In[4]:


# Unzip the Model and the Training Data

print('Unzipping...')
print('=' * 70)
get_ipython().system("unzip -j 'Training-Data/Rock-Piano-Training-Data.zip'")

get_ipython().system('cat Model/Rock-Piano-Trained-Model.zip* > Rock-Piano-Trained-Model.zip')
print('=' * 70)

get_ipython().system('unzip -j Rock-Piano-Trained-Model.zip')
print('=' * 70)
print('Done!')


# In[5]:


#@title Load/Reload the model
full_path_to_model_checkpoint = "Rock-Piano-Trained-Model.pth" #@param {type:"string"}

print('Loading the model...')
config = GPTConfig(VOCAB_SIZE, 
                   max_seq,
                   dim_feedforward=dim_feedforward,
                   n_layer=6, 
                   n_head=8, 
                   n_embd=512,
                   enable_rpr=True,
                   er_len=max_seq)

model = GPT(config).to(get_device())

model.load_state_dict(torch.load(full_path_to_model_checkpoint))

print('Done!')


# In[7]:


# Load the Training Data for priming the model

# Re-running this code will owerwrite the continuation code below

# Re-run this to prime from Training Data at any time

inputs = []

song_ints = []

data = TMIDIX.Tegridy_Any_Pickle_File_Reader('Rock-Piano-Training-Data')


pe = data[0]
for d in tqdm(data):
    song_ints.extend([int(abs(d[1] - pe[1]) / 10)+33, int(d[2] / 10)+33, d[3]+33, d[4]+33, 10])
    pe = d
    
print('Done!')


# In[9]:


# Run this code to prime with your own custom MIDI

# MIDI MUST HAVE ONLY PIANO-DRUMS instruments or it may not continue the composition properly

print('Loading your custom MIDI...')

data = TMIDIX.Optimus_MIDI_TXT_Processor('Rock-Piano-Continuation-Seed-1.mid', 
                                         MIDI_channel=16, 
                                         musenet_encoding=True, 
                                         perfect_timings=True)


pe = data[2][0]
for d in tqdm(data[2]):
    inputs.extend([int(abs(d[1] - pe[1]) / 10)+33, int(d[2] / 10)+33, d[3]+33, d[4]+33, 10])
    pe = d
    
print('Done!')


# In[11]:


#@title Generate Music

number_of_tokens_to_generate = 1024 #@param {type:"slider", min:8, max:1024, step:8}
use_random_primer = False #@param {type:"boolean"}
number_of_ticks_per_quarter = 500 #@param {type:"slider", min:50, max:1000, step:50}
dataset_time_denominator = 10
melody_conditioned_encoding = False
encoding_has_MIDI_channels = False 
encoding_has_velocities = False
simulate_velocity = True #@param {type:"boolean"}
save_only_first_composition = True
chars_encoding_offset_used_for_dataset = 33

fname = 'Rock-Piano-Composition'

print('Rock Piano Model Generator')

output_signature = 'Rock Piano'
song_name = 'RGA Composition'

model.eval()

if use_random_primer:
    sequence = [random.randint(10, 387) for i in range(64)]
    idx = secrets.randbelow(len(sequence))
    rand_seq = model.generate(torch.Tensor(sequence[idx:idx+120]), target_seq_length=number_of_tokens_to_generate)
    out = rand_seq[0].cpu().numpy().tolist()

else:
    out = []
  
    #try:
    if len(inputs) > 0:
      rand_seq = model.generate(torch.Tensor(inputs[-128:]), target_seq_length=number_of_tokens_to_generate)
      out = rand_seq[0].cpu().numpy().tolist()
    else:
      idx = secrets.randbelow(len(song_ints))
      rand_seq = model.generate(torch.Tensor(song_ints[idx:idx+120]), target_seq_length=number_of_tokens_to_generate)
      out = rand_seq[0].cpu().numpy().tolist()
  
  #except:
    #print('=' * 50)
    #print('Error! Try random priming instead!')
    #print('Shutting down...')
    #print('=' * 50)

if len(out) != 0:
  song = []
  sng = []
  for o in out:
    if o != 10:
      sng.append(o)
    else:
      if len(sng) == 4:
        song.append(sng)
      sng = []

  char_offset = 33
  song_f = []
  time = 0
  for s in song:
    
    song_f.append(['note', (abs(time)) * 10, (s[1]-char_offset) * 10, s[2]-char_offset, s[3]-char_offset, s[3]-char_offset])
    time += (s[0] - char_offset)

  detailed_stats = TMIDIX.Tegridy_SONG_to_MIDI_Converter(song_f,
                                                        output_signature = output_signature,  
                                                        output_file_name = fname, 
                                                        track_name=song_name, 
                                                        number_of_ticks_per_quarter=number_of_ticks_per_quarter)

  print('Done!')

  #print('Downloading your composition now...')
  #from google.colab import files
  #files.download(fname + '.mid')

  print('=' * 70)
  print('Detailed MIDI stats:')
  for key, value in detailed_stats.items():
        print('=' * 70)
        print(key, '|', value)

  print('=' * 70)

else:
  print('Models output is empty! Check the code...')
  print('Shutting down...')


# In[ ]:


#@title Auto-Regressive Generator

#@markdown NOTE: You much generate a seed composition first or it is not going to start

number_of_cycles_to_run = 5 #@param {type:"slider", min:1, max:50, step:1}
number_of_prime_tokens = 128 #@param {type:"slider", min:64, max:256, step:64}

print('=' * 70)
print('Rock Piano Auto-Regressive Model Generator')
print('=' * 70)
print('Starting up...')
print('=' * 70)
print('Prime length:', len(out))
print('Prime tokens:', number_of_prime_tokens)
print('Prime input sequence', out[-8:])

if len(out) != 0:
  print('=' * 70)
  out_all = []
  out_all.append(out)
  for i in tqdm(range(number_of_cycles_to_run)):
      rand_seq1 = model.generate(torch.Tensor(out[-number_of_prime_tokens:]), target_seq_length=1024)
      out1 = rand_seq1[0].cpu().numpy().tolist()
      out_all.append(out1[number_of_prime_tokens:])
      out = out1[number_of_prime_tokens:]
      
      print(chr(10))
      print('=' * 70)
      print('Block number:', i+1)
      print('Composition length so far:', (i+1) * 1024, 'tokens')
      print('=' * 70)

  print('Done!' * 70)
  print('Total blocks:', i+1)
  print('Final omposition length:', (i+1) * 1024, 'tokens')
  print('=' * 70)
  
  out2 = []
  for o in out_all:
    out2.extend(o)

  if len(out2) != 0:
    song = []
    sng = []
    for o in out2:
      if o != 10:
        sng.append(o)
      else:
        if len(sng) == 4:
          song.append(sng)
        sng = []

    char_offset = 33
    song_f = []
    time = 0
    for s in song:
      
        song_f.append(['note', (abs(time)) * 10, (s[1]-char_offset) * 10, s[2]-char_offset, s[3]-char_offset, s[3]-char_offset])      
        time += (s[0] - char_offset)

    song_name = 'Auto-Regressive RGA Composition'

    detailed_stats = TMIDIX.Tegridy_SONG_to_MIDI_Converter(song_f,
                                                          output_signature = output_signature,  
                                                          output_file_name = fname, 
                                                          track_name=song_name, 
                                                          number_of_ticks_per_quarter=number_of_ticks_per_quarter)

    print('Done!')

    #print('Downloading your composition now...')
    #from google.colab import files
    #files.download(fname + '.mid')

    print('=' * 70)
    print('Detailed MIDI stats:')
    for key, value in detailed_stats.items():
          print('=' * 70)
          print(key, '|', value)

    print('=' * 70)

else:
  print('=' * 70)
  print('INPUT ERROR !!!')
  print('Prime sequence is empty...')
  print('Please generate prime sequence and retry')

print('=' * 70)


# # Congrats! :) You did it! :)
