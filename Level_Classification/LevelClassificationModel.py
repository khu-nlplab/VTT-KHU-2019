import MemoryLevel
import LogicLevel
import time

def printPredictionLevel(question, clip_description, scene_description):
    memory_output = MemoryLevel.Memory_level_model(question, clip_description, scene_description)
    logic_output = LogicLevel.Logic_level_model(question, clip_description, scene_description)

    print('Memory Level : ', memory_output)
    print('Logic Level : ', logic_output)

    return memory_output, logic_output

if __name__ == "__main__":
    start_time = time.time()
    question = 'What is Chandler hopeful about with the unknown female'
    clip_description = 'Chandler is on the phone with a female and then Chandler and Ross talk about meeting her the next day'
    scene_description = ''
    printPredictionLevel(question, clip_description, scene_description)

    end_time=time.time()
    print('Process time: ', end_time-start_time)
