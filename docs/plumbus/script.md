This is the "End Game" for the data model. We are going to build a Python script that uses `pandas`, `random`, and `faker` to generate four distinct datasets (CSVs) that link together.

This script implements:

1.  **The "17-Count" Box Logic.**
2.  **The Time-Travel Shipping Bug.**
3.  **The Dual-Currency Chaos.**
4.  **The Nonsense Attribute Columns.**

### The Python Generator Script

You can run this locally. It generates `plumbus_products.csv`, `plumbus_inventory.csv`, `plumbus_transactions.csv`, and `dim_entities.csv`.

```python
import pandas as pd
import random
import uuid
from faker import Faker
from datetime import datetime, timedelta

# Initialize Faker for generic filler data (addresses/names)
fake = Faker

# --- CONFIGURATION: RICK & MORTY LORE ---

VENDORS = [
    "Ants in My Eyes Johnson Electronics", "Real Fake Doors", "Stealy's Procurement", 
    "Turbulent Juice Co.", "Curse Purge Plus", "Gearhead Parts", "Sanchez Smith Garage"
]

CUSTOMERS = [
    "The Galactic Federation", "Council of Ricks", "Mr. Poopybutthole", 
    "Krombopulos Michael", "Birdperson (Deceased)", "The President (USA)", 
    "Citadel Gift Shop", "Lil' Bits Franchise Owner"
]

LOCATIONS = [
    {"id": "DC-001", "name": "Citadel Central Hub", "type": "CDC"},
    {"id": "DC-137", "name": "Earth C-137 Garage", "type": "FSL"},
    {"id": "DC-GROM", "name": "Gromflomite Customs", "type": "Cross-Dock"},
    {"id": "DC-JERRY", "name": "Jerryboree Daycare", "type": "Dead_Stock"},
    {"id": "DC-VOID", "name": "Interdimensional Void", "type": "Lost_Goods"}
]

# --- 1. PRODUCT MASTER (PIM & SKUs) ---

def generate_products:
    print("Generating Plumbus PIM Data...")
    base_products = [
        ("The Plumbus (Classic)", "Standard_Pink", 15.5),
        ("Lil' Bits Tiny Plumbus", "Micro_Mini", 2.5),
        ("Plumbus Magnum", "Industrial_Grey", 450.0),
        ("Dark Plumbus", "Void_Black", 666.0),
        ("Defective Plumbus", "No_Schleem", 12.0),
        ("Invisible Plumbus", "Cloaked", 0.0)
    ]
    
    product_data = []
    
    for name, variant, weight in base_products:
        # Create the Single Unit
        sku_single = f"PLM-{variant[:3].upper}-001"
        product_data.append({
            "product_id": str(uuid.uuid4),
            "sku": sku_single,
            "name": name,
            "type": "Single_Unit",
            "pack_qty": 1,
            "texture": random.choice(["Grumbo_Smooth", "Rough", "Slime_Coated"]),
            "fleeb_viscosity": random.choice(["10W-30", "Gelatinous", "Aqueous"]),
            "schlami_ph": round(random.uniform(4.0, 6.0), 2),
            "weight_brapples": weight,
            "cost_flurbos": round(random.uniform(10, 50), 2),
            "is_cursed": random.choice([True, False, False, False]) # 25% chance of curse
        })
        
        # Create the Box of 17 (The Prime Number Headache)
        sku_box = f"PLM-{variant[:3].upper}-017"
        product_data.append({
            "product_id": str(uuid.uuid4),
            "sku": sku_box,
            "name": f"{name} - Prime Pack",
            "type": "Box_Config",
            "pack_qty": 17,
            "texture": "N/A (Outer_Carton)",
            "fleeb_viscosity": "N/A",
            "schlami_ph": 0.0,
            "weight_brapples": weight * 17,
            "cost_flurbos": round((random.uniform(10, 50) * 17) * 0.9, 2), # 10% bulk discount
            "is_cursed": False
        })

    return pd.DataFrame(product_data)

# --- 2. INVENTORY SNAPSHOT (ERP) ---

def generate_inventory(df_products):
    print("Generating Inventory Snapshots...")
    inventory_data = []
    
    skus = df_products['sku'].tolist
    
    for loc in LOCATIONS:
        for sku in skus:
            # Randomize logic: Some places stock everything, some have nothing
            if random.random > 0.3: 
                qty = random.randint(0, 5000)
                
                # Logic: If it's the Void, inventory is negative
                if loc['id'] == "DC-VOID":
                    qty = qty * -1
                
                # Logic: Earth C-137 has low stock (Rick uses it)
                if loc['id'] == "DC-137":
                    qty = random.randint(0, 10)

                inventory_data.append({
                    "snapshot_date": datetime.now.date,
                    "location_id": loc['id'],
                    "sku": sku,
                    "qty_on_hand": qty,
                    "qty_reserved": int(qty * 0.1), # 10% reserved
                    "bin_location": f"Sector-{random.randint(1,9)}-{random.choice(['A','B','C'])}"
                })
    
    return pd.DataFrame(inventory_data)

# --- 3. TRANSACTIONS (SALES & MVMT) ---

def generate_transactions(df_products, num_rows=1000):
    print("Generating Chaos Transactions...")
    data = []
    skus = df_products['sku'].tolist
    
    for _ in range(num_rows):
        order_date = fake.date_between(start_date='-1y', end_date='today')
        sku = random.choice(skus)
        qty = random.randint(1, 100)
        
        # TIME TRAVEL LOGIC: 5% chance ship date is BEFORE order date
        if random.random < 0.05:
            ship_date = order_date - timedelta(days=random.randint(1, 50))
            status_note = "TEMPORAL_PARADOX_DETECTED"
        else:
            ship_date = order_date + timedelta(days=random.randint(1, 5))
            status_note = "Standard_Portal_Delivery"

        # CURRENCY LOGIC
        currency = "Flurbos"
        price = random.uniform(20, 100)
        if random.random < 0.2: # 20% sales in Schmeckles
            currency = "Schmeckles"
            price = price / 148 # Exchange rate
            
        data.append({
            "trans_id": f"TX-{fake.hexify(text='^^^^-^^^^')}",
            "customer": random.choice(CUSTOMERS),
            "order_date": order_date,
            "ship_date": ship_date,
            "sku": sku,
            "qty": qty,
            "total_amount": round(price * qty, 2),
            "currency": currency,
            "shipping_status": status_note,
            "carrier": random.choice(["Federation Logistics", "Rick's Ship", "Heistotron"])
        })
        
    return pd.DataFrame(data)

# --- EXECUTION ---

# 1. Generate Products
df_products = generate_products
df_products.to_csv("plumbus_pim_master.csv", index=False)

# 2. Generate Inventory
df_inventory = generate_inventory(df_products)
df_inventory.to_csv("plumbus_erp_inventory.csv", index=False)

# 3. Generate Transactions
df_sales = generate_transactions(df_products, num_rows=500)
df_sales.to_csv("plumbus_sales_history.csv", index=False)

# 4. Generate Entities (Dim Table)
df_entities = pd.DataFrame(LOCATIONS)
df_entities.to_csv("dim_locations.csv", index=False)

print("SUCCESS: Interdimensional Data Generated.")
print(df_products[['sku', 'name', 'pack_qty']].head(10))
```

-----

### What this data looks like (Preview)

When you run this, you will get files populated with data that looks like this:

**From `plumbus_pim_master.csv`:**
| sku             | name                               | pack\_qty | texture             | fleeb\_viscosity | schlami\_ph |
| :-------------- | :--------------------------------- | :-------- | :------------------ | :--------------- | :---------- |
| **PLM-STA-001** | The Plumbus (Classic)              | 1         | Slime\_Coated       | Gelatinous       | 5.42        |
| **PLM-STA-017** | The Plumbus (Classic) - Prime Pack | **17**    | N/A (Outer\_Carton) | N/A              | 0.0         |
| **PLM-DAR-001** | Dark Plumbus                       | 1         | Rough               | 10W-30           | 4.11        |

**From `plumbus_sales_history.csv` (Note the Paradox):**
| trans\_id   | customer              | order\_date    | ship\_date     | sku             | status\_note                    |
| :---------- | :-------------------- | :------------- | :------------- | :-------------- | :------------------------------ |
| TX-A9B1     | Galactic Federation   | 2024-03-10     | 2024-03-12     | PLM-STA-001     | Standard\_Portal\_Delivery      |
| **TX-Z009** | **Mr. Poopybutthole** | **2024-05-20** | **2024-04-15** | **PLM-LIL-001** | **TEMPORAL\_PARADOX\_DETECTED** |

### Next Step

You now have the raw files.

Would you like to design a **SQL Challenge / Quiz** based on this data? (e.g., *"Write a query to find all Plumbuses shipped before they were ordered, converted to Schmeckles"*).