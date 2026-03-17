This is a great constraint because "17" is a prime number, making it a logistic nightmare for palletizing—perfect for this universe.

Here is your **Master SKU List**, divided into **Singles (Consumer Units)** and **Distribution Packs (17-Count)**.

### 1\. The Packaging Hierarchy (The "17" Logic)

We need to define the packaging levels in your system so the data looks "enterprise."

  * **Level 1: The Single**
      * *Unit Name:* `Unit_Alpha`
      * *Qty:* 1
  * **Level 2: The Box**
      * *Unit Name:* `Gromflomite_Crate`
      * *Qty:* 17 Singles (The "Prime Batch")
  * **Level 3: The Pallet**
      * *Unit Name:* `Teleport_Pad_Stack`
      * *Qty:* 289 Singles (17 Crates of 17) - *Squared primes for maximum inefficiency.*

-----

### 2\. The Master SKU List

Here are 12 specific SKUs you can load into the database. I have included the "Standard" lineup plus some bizarre variants.

| SKU ID          | Product Name                  | Variant Type     | Packaging      | Description / Data Note                                    |
| :-------------- | :---------------------------- | :--------------- | :------------- | :--------------------------------------------------------- |
| **PLM-STD-001** | The Plumbus (Classic)         | `Standard_Pink`  | Single         | The original. Indeterminate functionality.                 |
| **PLM-STD-017** | The Plumbus (Classic) - Batch | `Standard_Pink`  | **Box of 17**  | 17 loose Plumbuses in a cardboard void.                    |
| **PLM-LIL-001** | Lil' Bits Tiny Plumbus        | `Micro_Mini`     | Single         | Fits in your mouth (don't put it in your mouth).           |
| **PLM-LIL-017** | Lil' Bits Tiny Plumbus - Sack | `Micro_Mini`     | **Sack of 17** | Sold exclusively at Lil' Bits.                             |
| **PLM-IND-900** | Plumbus Magnum (Industrial)   | `Grumbo_Heavy`   | Single         | Requires Level 4 clearance to lift. 800 Brapples.          |
| **PLM-DRK-666** | Dark Plumbus                  | `Void_Black`     | Single         | Absurdly heavy. Emits low hum.                             |
| **PLM-ERR-404** | "Oops\! All Dinglebop"        | `Defect_Variant` | Single         | Factory error. Just a stick. No Schleem.                   |
| **PLM-TRB-X10** | Turbulent Plumbus             | `Viscous_Flow`   | Single         | Coated in Turbulent Juice. Highly illegal in Sector C.     |
| **PLM-DIY-KIT** | IKEA Plumbus (Some Assembly)  | `Unassembled`    | Box            | Includes Dinglebop and a bucket of raw Schleem. Good luck. |
| **PLM-GLD-001** | Plumbus Prime (Gold Edition)  | `Luxury_Foil`    | Single         | Sold at Blitz and Chitz for 10,000 tickets.                |
| **PLM-REF-000** | Refurbished Plumbus           | `Used_Grade_B`   | Single         | "Stealy's Special." Slight Fleeb decay.                    |
| **PLM-COV-019** | Invisible Plumbus             | `Cloaked`        | Single         | We think it's in the box? We can't tell.                   |

-----

### 3\. Competitor / "White Label" Data

To make the dataset realistic, you need "dirty data" from third-party sellers who mess up your clean formatting.

  * **Vendor:** *Real Fake Doors*
      * **SKU:** `RFD-PLUM-FAKE`
      * **Name:** "Door-Stop Plumbus"
      * **Note:** It doesn't function; it just holds doors open.
  * **Vendor:** *Curse Purge Plus*
      * **SKU:** `CPP-DEMON-PLM`
      * **Name:** "Cursed Plumbus"
      * **Note:** Free, but you pay with your soul.

-----

### 4\. Updated Data Model integration

To handle the "17 to a box" logic, we simply add a `packaging_config` object to our JSON model:

```json
{
  "sku": "PLM-STD-017",
  "name": "The Plumbus (Classic) - Batch",
  "packaging_config": {
    "is_base_unit": false,
    "contains_sku": "PLM-STD-001",
    "quantity_contained": 17,
    "packaging_material": "Compressed_Dark_Matter_Cardboard",
    "dimensions": {
      "length": "17.5 Schmeckles",
      "width": "17.5 Schmeckles",
      "height": "17.5 Schmeckles"
    }
  }
}
```

### Next Step

We have the **Items** and the **Packaging Logic**.

Shall we move to the **Distribution Logic** (generating the "Ship-To" addresses using planet names like Gazorpazorp, Bird World, and Squanch Planet), or do you want to start building the **Python Script** to generate the actual CSV/JSON files?