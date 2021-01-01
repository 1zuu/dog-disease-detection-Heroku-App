import os
seed = 42
port = 5000

precausion_path = 'data/precausions.csv'
model_converter = "data/model.tflite"

host = '0.0.0.0'
all_symtoms =  ['Acute blindness', 'Urine infection', 'Red bumps', 'Loss of Fur', 'Licking', 'Grinning appearance', 'Coughing', 'Eye Discharge', 'Seizures', 'excess jaw tone', 'Coma', 'Weakness', 'Wounds', 'Neurological Disorders', 'blood in stools', 'Stiff and hard tail', 'Dry Skin', 'Lameness', 'Swelling of gum', 'Fever', 'Bloated Stomach', 'Face rubbing', 'Aggression', 'Wrinkled forehead', 'Lumps', 'Plaque', 'Blindness', 'Weight Loss', 'Swollen Lymph nodes', 'Excessive Salivation', 'Loss of Consciousness', 'Tender abdomen', 'Purging', 'Dandruff', 'Loss of appetite', 'Pale gums', 'Collapse', 'Constipation', 'Hunger', 'Discomfort', 'Pain', 'Paralysis', 'Red patches', 'Fur loss', 'Losing sight', 'WeightLoss', 'Sepsis', 'Increased drinking and urination', 'Bad breath', 'Itchy skin', 'Receding gum', 'Irritation', 'Enlarged Liver', 'Eating grass', 'Nasal Discharge', 'Depression', 'lethargy', 'Stiffness of muscles', 'Eating less than usual', 'Scratching', 'Severe Dehydration', 'Tartar', 'Cataracts', 'Swelling', 'Redness of gum', 'Diarrhea', 'Scabs', 'Breathing Difficulty', 'Difficulty Urinating', 'Continuously erect and stiff ears', 'Glucose in urine', 'Burping', 'Passing gases', 'Vomiting', 'Blood in urine', 'Smelly', 'Redness around Eye area', 'Bleeding of gum', 'Bloody discharge', 'Redness of skin', 'Lethargy', 'Abdominal pain', 'Lack of energy', 'Anorexia', 'Heart Complication', 'Yellow gums']
all_diseases = ['hepatitis ', 'diabetes', 'cancers', 'allergies', 'tetanus ', 'gingitivis', 'skin rashes', 'distemper', 'parvovirus', 'chronic kidney disease ', 'tick fever', 'gastrointestinal disease']
all_symtoms = list(map(str.lower,all_symtoms))
all_diseases = list(map(str.lower,all_diseases))
