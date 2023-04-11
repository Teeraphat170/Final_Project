def ToFirebase(namefile,okng,timeX,prediction_proba):
    
    from firebase import firebase

    firebaseX = firebase.FirebaseApplication("https://internship-project-swu-default-rtdb.firebaseio.com/",None)

    test = {'Time':timeX,
            'Name':namefile,
            'Result':okng,
            'Probability(OK)':prediction_proba[0][1],
            'Probability(NG)':prediction_proba[0][0]}

    result = firebaseX.put('SWUProject',namefile,test)
    return result
    
# For_Test
# namefile = 'Test1'
# okng = 'OK'
# timeX = '000000'
# print(ToFirebase(namefile,okng,timeX))

