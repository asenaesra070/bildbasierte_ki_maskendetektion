import os                                                # Automatische Ordnerverwaltung
import shutil                                            # Das Modul shutil bietet eine Reihe von High-Level-Operationen für Dateien und Dateisammlungen

#--> input folder : input_dir 2mal drin - alt copy
#--> output folder : output_dir - neu paste
def flatten_image_directory(root_dir):
    count = 0
    for class_name in ["with_mask", "without_mask"]:
        class_path = os.path.join(root_dir, class_name)
        # without_mask_flat oder with_mask_flat
        new_dir = os.path.join(root_dir, f"{class_name}_flat")
        os.makedirs(new_dir, exist_ok=True)
        # Ich habe die Namen der Dateien vor die Bilder geschrieben, um anzuzeigen, woher sie stammen.
        for person_folder in os.listdir(class_path):
            person_path = os.path.join(class_path, person_folder)
            if os.path.isdir(person_path):
                for filename in os.listdir(person_path):
                    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                        src = os.path.join(person_path, filename)
                        new_name = f"{person_folder}_{filename}"
                        dst = os.path.join(new_dir, new_name)
                        shutil.copy2(src, dst)
                        count += 1
    print(f"Insgesamt wurden {count} Bilder verschoben und umbenannt.")

# Path vom Images in data Folder
flatten_image_directory(r"C:\Users\guler\OneDrive - Hochschule Düsseldorf\Desktop\BildBasierteKI\bildbasierte_ki_maskendetektion\data\RWMFD")


# Am Ende:
#    --> with_mask  ! löscht
#    --> with_mask_flat
#    --> without_mask   ! löscht
#    --> without_mask_flat