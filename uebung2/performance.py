import matplotlib
matplotlib.use('Agg')  # MUSS VOR pyplot importiert werden! Prevents Qt/xcb crashes.
import matplotlib.pyplot as plt

# Hier kommen deine Daten hin
nx = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152]

# Daten für 1 Thread
no_blocking_1t = [1631.8, 1646.5, 1593.1, 1666.9, 1671.0, 1634.2, 1517.5, 1493.7, 1474.3, 1479.5, 1449.7, 1340.1, 1064.3, 1020.6]
blocked_1t     = [1658.1, 1674.6, 1616.8, 1683.2, 1685.2, 1644.6, 1494.6, 1470.8, 1478.2, 1510.3, 1520.7, 1508.4, 1474.2, 1384.1]

# Daten für 2 Threads
no_blocking_2t = [1808.7, 1817.1, 1774.5, 1856.4, 1855.6, 1826.3, 1662.0, 1660.3, 1654.0, 1630.8, 1497.4, 1229.8, 1108.1, 1111.7]
blocked_2t     = [1806.5, 1817.9, 1769.0, 1853.4, 1855.2, 1824.6, 1678.9, 1649.4, 1659.9, 1669.4, 1669.9, 1650.5, 1631.1, 1543.6]

# Daten für 4 Threads
no_blocking_4t = [1781.4, 1763.7, 1727.0, 1793.0, 1788.4, 1779.5, 1598.7, 1609.6, 1579.4, 1473.9, 1130.5, 1052.9, 1067.2, 1140.6]
blocked_4t     = [1773.4, 1760.8, 1733.6, 1790.5, 1791.4, 1779.2, 1657.8, 1621.9, 1625.9, 1645.4, 1640.2, 1604.3, 1567.4, 1511.3]

# Daten für 8 Threads
no_blocking_8t = [1653.3, 1569.3, 1568.3, 1585.8, 1587.9, 1571.0, 1493.5, 1469.5, 1393.3, 1127.0,  985.1,  924.1, 1060.4, 1414.6]
blocked_8t     = [1671.0, 1565.0, 1569.6, 1579.4, 1585.7, 1565.9, 1557.8, 1539.7, 1547.1, 1554.7, 1496.1, 1490.3, 1493.5, 1442.7]

plt.figure(figsize=(11, 7))

# Kurven zeichnen (Beispielhaft für 1, 2, 4 und 8 Threads)
colors = {1: '#1f77b4', 2: '#ff7f0e', 4: '#2ca02c', 8: '#d62728'}

for t, nb, bl in [(1, no_blocking_1t, blocked_1t), 
                  (2, no_blocking_2t, blocked_2t), 
                  (4, no_blocking_4t, blocked_4t), 
                  (8, no_blocking_8t, blocked_8t)]:
    plt.plot(nx, nb, marker='o', label=f'{t} Thread(s) - No Blocking', color=colors[t])
    plt.plot(nx, bl, marker='s', linestyle='--', label=f'{t} Thread(s) - Blocked', color=colors[t])

# Achsen konfigurieren
plt.xscale('log', base=2)
plt.xticks(nx, [str(x) for x in nx], rotation=45)
plt.xlabel('Row Length $N_x$ (logarithmic scale)')
plt.ylabel('Performance (Million Updates / s)')
plt.title('Jacobi Sweep Performance & Spatial Blocking Scaling')
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

plt.tight_layout()

# WICHTIG: Kein plt.show() am Ende! Nur speichern!
plt.savefig('jacobi_performance.png', dpi=300)
print("Diagramm erfolgreich als 'jacobi_performance.png' gespeichert!")
