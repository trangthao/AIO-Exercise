def evaluate_classification_f1score(tp, fp, fn):
    # Check if the inputs are integers
    if not isinstance(tp, int):
        print("tp must be int")
        return
    if not isinstance(fp, int):
        print("fp must be int")
        return
    if not isinstance(fn, int):
        print("fn must be int")
        return
    
    # Check if the inputs are greater than zero
    if tp <= 0 or fp <= 0 or fn <= 0: 
        print("tp and fp and fn must be greater than zero")
        return
    
    # Calculate Precision, Recall, and F1-Score
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)

    # Print the results
    print(f"precision is {precision}")
    print(f"recall is {recall}")
    print(f"f1-score is {f1_score}")

# Examples
evaluate_classification_f1score(tp=2, fp=3, fn=4)
evaluate_classification_f1score(tp='a', fp=3, fn=4)
evaluate_classification_f1score(tp=2, fp='a', fn=4)
evaluate_classification_f1score(tp=2, fp=3, fn='a')
evaluate_classification_f1score(tp=2, fp=3, fn=0)
evaluate_classification_f1score(tp=2.1, fp=3, fn=0)