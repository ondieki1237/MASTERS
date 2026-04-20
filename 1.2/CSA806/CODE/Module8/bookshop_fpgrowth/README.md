# Bookshop FP-Growth Project

## Files
- [bookshop_fp_growth.ipynb](bookshop_fp_growth.ipynb) — Spark notebook for synthetic data generation, FP-Growth, and rule extraction
- [report.md](report.md) — written report with findings and recommendations

## How to Run
1. Install PySpark in your Python environment.
2. Open the notebook in VS Code or Jupyter.
3. Run the notebook cells in order.
4. Review the printed frequent itemsets and association rules.
5. Check the exported parquet outputs:
   - `bookshop_freq_itemsets`
   - `bookshop_association_rules`

## Project Focus
This project uses a **bookshop retail scenario** with items such as:
- books
- pens
- notebooks
- bookmarks
- highlighters
- planners
- sticky notes
- diaries
- textbooks

## Expected Insights
- Strong co-purchase patterns between stationery items.
- Useful cross-sell opportunities near books and reading accessories.
- Better stock planning for student and office-related products.
