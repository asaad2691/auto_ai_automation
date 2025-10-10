import json
import random

def start_game():
    with open('questions.json') as f:
        data = json.load(f)
    
    score = 0
    total_ques = len(data['Questions'])
  
    for i in range(total_ques):
        print("\nQuestion", str(i+1), "out of", total_ques )
        question = data['Questions'][i]
        print("  ", question['question'])
        
        options = question['options']
        random.shuffle(options)
      
        for j in range(4):
            print(chr(65 + j), ") ", options[j])
          
        user_answer = input("\nEnter your choice: ").upper()
        
        if user_answer == question['correct']:
            score += 1
            print("Correct!")
        else: 
            print("Wrong. The correct answer was", question['correct'])
      
    print("\nYour total score is", score, "out of", total_ques)
    
if __name__ == '__main__':
    start_game()
