import shutil
import os
shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results_final', 'MATHIR_FINAL_REPORT.html'), os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'reports', 'MATHIR_FINAL_REPORT.html'))
size = os.path.getsize(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'reports', 'MATHIR_FINAL_REPORT.html'))
print(f'HTML report: {size/1024:.1f} KB')