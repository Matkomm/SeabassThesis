import matplotlib.pyplot as plt

# Data
years = list(range(1980, 2022))
harvest = [
    130, 165, 230, 300, 360, 581, 949, 1151, 1667, 2245, 3921, 6744, 10139, 
    16811, 17092, 22263, 26320, 33832, 43804, 53898, 70694, 60385, 58133, 
    73456, 74064, 95044, 98173, 104474, 115453.685, 112599.233, 134328.399, 
    137276.156, 146252.405, 147202.194, 155924.066, 163439.163, 191509.923, 
    215571.884, 235722.377, 264030.531, 277878.226, 299809.871
]

# Plot
plt.figure(figsize=(10, 6))
#plt.plot(years, harvest, marker='o', linestyle='-')
plt.bar(years[:16], harvest[:16], color='skyblue')
plt.title('European Seabass Production (1980-1995)')
plt.xlabel('Year')
plt.ylabel('Harvest (Tonnes)')
#plt.grid(True)
plt.grid(axis='y')
plt.xticks(range(1980, 1996, 5))
plt.tight_layout()
plt.show()
