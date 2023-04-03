# /usr/bin/env python
# -*- coding: UTF-8 -*-

#====setting for PDP attack======

Use_GPU_id='0'

Attack_data='mr'  # mr imdb
Attack_model='wordLSTM'  # wordLSTM bert


result_filename='FindAdresult/'+Attack_data+'_'+Attack_model+'_'+'onlyDP2.3'
#result_filename=None

write_mod='w'    #'a'

attack_data_num=200     #how many texts attack?
attack_start_textid=0     #attack start_id


filt_text_length=10000

TOPK_bach=32 # 128*5