import MemoryLevel
import LogicLevel
import time

def model_load(memory_check_point, logic_check_point):
    Memory_model = MemoryLevel.Memory_level_model_init(memory_check_point)
    Logic_model = LogicLevel.Logic_level_model_init(logic_check_point)

    return Memory_model, Logic_model

def printPredictionLevel(question, clip_description, scene_description, Logic_check_point, Memory_check_point):

    memory_output = MemoryLevel.Memory_level_model(question, clip_description, scene_description, Memory_check_point)
    logic_output = LogicLevel.Logic_level_model(question, clip_description, scene_description, Logic_check_point)

    print('Memory Level : ', memory_output)
    print('Logic Level : ', logic_output)

    return memory_output, logic_output

if __name__ == "__main__":
    Logic_check_point = 'model/Logic_model.bin'
    Memory_check_point = 'model/Memory_model.bin'

    Memory_model, Logic_model = model_load(Memory_check_point, Logic_check_point)

    start_time = time.time()
    question = 'What is Chandler hopeful about with the unknown female'
    clip_description = 'Chandler is on the phone with a female and then Chandler and Ross talk about meeting her the next day'
    scene_description = "Chandler pretended to be Bob to set up a date with a lady. Chandler's plan is to sit at the table next to the lady and wait for her to come to him for comfort when she realizes Bob won't show up. Ross and Chandler both listen to the answering machine. Chandler plans to impersonate Bob., Chandler is on the phone with a female and then Chandler and Ross talk about meeting her the next day."

    printPredictionLevel(question, clip_description, scene_description, Logic_model, Memory_model)

    end_time=time.time()

    print('Process time: ', end_time-start_time)
