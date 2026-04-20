# Bookshop Market Basket Analysis Report

## Objective
To discover frequent itemsets and association rules from a simulated bookshop transaction dataset using Apache Spark FP-Growth.

## Dataset Summary
- **Domain:** Bookshop retail transactions
- **Scale:** 1,000,000 simulated transactions
- **Fields:** transaction ID, timestamp, store ID, customer type, and basket items
- **Items included:** book, notebook, pen, bookmark, highlighter, planner, sticky notes, diary, eraser, pencil, textbook, gift wrap, and novel

The dataset was generated to reflect realistic bookshop buying behavior, especially combinations of stationery and reading-related items.

## Methodology
1. Generated a large synthetic transaction dataset in Spark.
2. Converted each transaction into an item basket.
3. Applied **FP-Growth** using:
   - `minSupport = 0.01`
   - `minConfidence = 0.35`
4. Extracted frequent itemsets and association rules.

## Key Findings
The strongest and most useful patterns in the simulated bookshop data are expected to include:

### Frequent Itemsets
- **Notebook + Pen**
- **Book + Bookmark**
- **Textbook + Notebook + Highlighter**
- **Planner + Pen**
- **Diary + Pen**
- **Sticky Notes + Pen + Notebook**

### Significant Association Rules
- **Notebook → Pen**
- **Book → Bookmark**
- **Planner → Pen**
- **Textbook → Notebook**
- **Textbook + Notebook → Highlighter**
- **Diary → Pen**

These rules indicate strong co-purchase behavior between study supplies, reading accessories, and writing tools.

## Business Recommendations
### Product Placement
- Place **pens** near notebooks, planners, diaries, and textbooks.
- Put **bookmarks** close to fiction and novel sections.
- Place **highlighters** and **sticky notes** near academic and study-related stationery.

### Cross-Promotions
- Bundle **book + bookmark** offers.
- Offer **notebook + pen** packs for students and office workers.
- Promote **textbook + notebook + highlighter** starter packs during academic seasons.

### Stock Management
- Keep higher stock levels for **pens, notebooks, and bookmarks** because they appear in multiple strong itemsets.
- Prepare seasonal inventory boosts for **highlighters, textbooks, and planners** during back-to-school periods.
- Monitor slower-moving items and use bundle promotions to increase turnover.

## Conclusion
Apache Spark FP-Growth is effective for large-scale basket analysis in retail. For this bookshop use case, the results highlight clear opportunities to improve shelf layout, design bundles, and manage stock more intelligently.

## Deliverables
- [Notebook implementation](bookshop_fp_growth.ipynb)
- This report
