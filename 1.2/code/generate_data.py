# generate_data.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

counties = ["Nairobi", "Mombasa", "Kisumu", "Nakuru"]
diseases = ["Malaria", "Typhoid", "Cholera", "TB"]

data = []

for i in range(10000):
    data.append({
        "patient_id": f"P{i}",
        "hospital": f"Hospital_{random.randint(1,5)}",
        "county": random.choice(counties),
        "visit_date": datetime.today() - timedelta(days=random.randint(0, 60)),
        "disease": random.choice(diseases),
        "diagnosis_code": f"D{random.randint(100,999)}"
    })

df = pd.DataFrame(data)
df.to_csv("patient_raw.csv", index=False)

print("Raw dataset created")
