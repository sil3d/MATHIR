import shutil
import os
shutil.copy('D:/SECRET_PROJECT/MATHIR/benchmarks/results_final/MATHIR_FINAL_REPORT.html', 'D:/SECRET_PROJECT/MATHIR/benchmarks/MATHIR_FINAL_REPORT.html')
size = os.path.getsize('D:/SECRET_PROJECT/MATHIR/benchmarks/MATHIR_FINAL_REPORT.html')
print(f'HTML report: {size/1024:.1f} KB')