import matplotlib
matplotlib.use('Agg')  # Verhindert den Qt/xcb-Fehler auf der Konsole
import matplotlib.pyplot as plt

# Die Zeilenlängen Nx aus dem Testlauf
nx = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152]

# Deine Messwerte aus dem ALLERERSTEN Durchlauf (sequentiell / 1 Thread)
no_blocking_1t = [1631.8, 1646.5, 1593.1, 1666.9, 1671.0, 1634.2, 1517.5, 1493.7, 1474.3, 1479.5, 1449.7, 1340.1, 1064.3, 1020.6]
blocked_1t     = [1658.1, 1674.6, 1616.8, 1683.2, 1685.2, 1644.6, 1494.6, 1470.8, 1478.2, 1510.3, 1520.7, 1508.4, 1474.2, 1384.1]

plt.figure(figsize=(10, 6))

# Zeichne die beiden Kurven für den sequentiellen Vergleich
plt.plot(nx, no_blocking_1t, marker='o', label='Jacobi Sweep - No Blocking', color='#d62728', linestyle='-')
plt.plot(nx, blocked_1t, marker='s', label='Jacobi Sweep - Blocked (B=10k)', color='#1f77b4', linestyle='--')

# Schichtbedingung als vertikale Hilfslinie (Theoretisches L3-Limit bei ca. 524k)
plt.axvline(x=524288, color='green', linestyle=':', label='Theoretical L3 Limit (524,285)')

# Achsen- und Diagramm-Design
plt.xscale('log', base=2)
plt.xticks(nx, [str(x) for x in nx], rotation=45)
plt.xlabel('Row Length $N_x$ (logarithmic scale, base 2)', fontsize=11)
plt.ylabel('Performance [MUPs/s]', fontsize=11)
plt.title('Jacobi Sweep Performance vs. Row Length $N_x$ (1 Thread)', fontsize=13, fontweight='bold')
plt.grid(True, which="both", ls=":", alpha=0.5)
plt.legend(loc="lower left")

plt.tight_layout()

# Speicher den Plot als PNG
plt.savefig('jacobi_performance_1t.png', dpi=300)
print("Diagramm erfolgreich als 'jacobi_performance_1t.png' gespeichert!")
