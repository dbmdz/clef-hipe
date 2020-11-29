import sys

filename = sys.argv[1] # like: HIPE-data-v1.2-train-de-normalized-manual-eos.tsv
export_dir = sys.argv[2] # like: training-v1.2/de/manual_corrected

all_documents = []
current_document = []

with open(filename, "rt") as f_p:
    for line in f_p:
        line = line.rstrip()

        if line.startswith("TOKEN"):
            continue

        if line.startswith("# language = de"):
            # New document started

            if current_document:
                all_documents.append(current_document)
                current_document = []
            
            current_document.append(line)
            continue

        current_document.append(line)
    
    if current_document:
        all_documents.append(current_document)
        current_document = []

for document in all_documents:
    #print(document[3])
    document_filename = document[3].split("=")[1].strip() + ".txt"
    
    with open(f"{export_dir}/{document_filename}", "wt") as f_p:
        f_p.writelines("\n".join(document))
        f_p.write("\n\n")
