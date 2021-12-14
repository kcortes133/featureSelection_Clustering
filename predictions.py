def predict(X, y, model):
    right = 0
    wrong = 0
    fpos = 0
    fneg = 0
    predicts = model.predict(X)
    for i,j in zip(predicts,y):
        if i == j:
            right+=1
        else:
            wrong+=1
        if i == 0 and j == 1:
            # false pos
            fpos +=1
        elif i == 1 and j == 0:
            # false neg
            fneg +=1
    print(fpos, fneg)

    return
