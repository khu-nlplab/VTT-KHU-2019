import MemoryLevel
import LogicLevel
import time

def printPredictionLevel(text_a, text_b):
    memory_output = MemoryLevel.Memory_level_model(text_a,text_b)
    logic_output = LogicLevel.Logic_level_model(text_a, text_b)

    print('Memory Level : ', memory_output)
    print('Logic Level : ', logic_output)

    return memory_output, logic_output

start_time = time.time()
text_a = 'What is Chandler hopeful about with the unknown female?'
text_b = 'Chandler is on the phone with a female and then Chandler and Ross talk about meeting her the next day.'
printPredictionLevel(text_a, text_b)
end_time=time.time()
print('Process time: ', end_time-start_time)
