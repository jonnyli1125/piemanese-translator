with open('replacements.csv', 'r', encoding='utf-8') as f:
    lines = f.readlines()

with open('replacements.csv', 'w', encoding='utf-8') as f:
    f.writelines(sorted(line.lower() for line in lines if '\t' in line))
