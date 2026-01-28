options(warn=-1)
options(scipen=999)

# Load libraries
library(tidyverse)
library(ggplot2)
library(plotly)
library(htmltools)
library(scales)
library(lubridate)
library(jsonlite)

# Create temp_charts directory if it doesn't exist
if (!dir.exists("temp_charts")) {
  dir.create("temp_charts")
}

# Read the dataset
df <- read.csv('C:/Users/admin/Desktop/R Automation/temp_data/source.csv', stringsAsFactors=FALSE, fileEncoding="UTF-8")

# 2. DATA PREPARATION
# Ensure 'quantity_ordered' and 'price' are numeric
df$quantity_ordered <- as.numeric(df$quantity_ordered)
df$price <- as.numeric(df$price)

# Date Parsing for 'order_date' - handle multiple formats
# Sample data shows 'DD-MM-YYYY' and 'M/D/YYYY'
df$order_date_parsed <- parse_date_time(df$order_date, orders=c("dmy", "mdy"))

# Filter out rows where order_date could not be parsed (if any)
df <- df %>% filter(!is.na(order_date_parsed))

# Extract month name and month number for ordering
df$month <- format(df$order_date_parsed, "%B")
df$month_num <- format(df$order_date_parsed, "%m")

# Clean 'city' column by trimming leading/trailing whitespace
df$city <- trimws(df$city)

# Calculate total revenue for each order line item
df$total_revenue <- df$quantity_ordered * df$price

# 3. SMART KPI CONFIGURATION
kpi_config <- list(
  slicers = c("city", "product_type"),
  kpis = list(
    list(label = "Total Revenue", formula = "quantity_ordered * price", agg = "sum", fmt = "$"),
    list(label = "Total Orders", formula = "order_id", agg = "count_distinct", fmt = "num"),
    list(label = "Avg Order Value", formula = "quantity_ordered * price", agg = "mean", fmt = "$") # Mean revenue per line item
  )
)
write_json(kpi_config, "temp_charts/kpi_config.json", auto_unbox = TRUE)


# 4. TEXT INSIGHTS

# Best month for sales
monthly_sales <- df %>%
  group_by(month, month_num) %>%
  summarise(TotalRevenue = sum(total_revenue, na.rm = TRUE)) %>%
  arrange(desc(TotalRevenue))

best_month <- monthly_sales %>% slice(1)

# City with highest total revenue
city_revenue <- df %>%
  group_by(city) %>%
  summarise(TotalRevenue = sum(total_revenue, na.rm = TRUE)) %>%
  arrange(desc(TotalRevenue))

top_city <- city_revenue %>% slice(1)

# Most frequently purchased products (top 5 for text output)
product_frequency <- df %>%
  group_by(product) %>%
  summarise(Count = n()) %>%
  arrange(desc(Count)) %>%
  slice_head(n = 5)

# Top 5 products contributing to sales (revenue)
top_5_products_revenue <- df %>%
  group_by(product) %>%
  summarise(TotalRevenue = sum(total_revenue, na.rm = TRUE)) %>%
  arrange(desc(TotalRevenue)) %>%
  slice_head(n = 5)

cat("--- Sales Performance Insights ---\n")
cat(sprintf("The best month for sales was '%s' with a total revenue of %s.\n",
            best_month$month, dollar(best_month$TotalRevenue, prefix = "$", accuracy = 1)))
cat(sprintf("The city that generated the highest total revenue was '%s' with %s.\n\n",
            top_city$city, dollar(top_city$TotalRevenue, prefix = "$", accuracy = 1)))

cat("Top 5 products most frequently purchased:\n")
for (i in 1:nrow(product_frequency)) {
  cat(sprintf("- %s (%s purchases)\n", product_frequency$product[i], comma(product_frequency$Count[i])))
}
cat("\n")

cat("Top 5 products contributing to sales are:\n")
for (i in 1:nrow(top_5_products_revenue)) {
  cat(sprintf("- %s (%s)\n", top_5_products_revenue$product[i], dollar(top_5_products_revenue$TotalRevenue[i], prefix = "$", accuracy = 1)))
}
cat("\n")


# 5. VISUALIZATION LOGIC
plots_list <- list()

# Plot 1: Monthly Sales Revenue
monthly_sales_ordered <- df %>%
  group_by(month, month_num) %>%
  summarise(TotalRevenue = sum(total_revenue, na.rm = TRUE)) %>%
  mutate(month_ordered = factor(month, levels = month.name[order(unique(month_num))])) %>% # Order months chronologically
  arrange(month_num)

p1 <- ggplot(monthly_sales_ordered, aes(x = month_ordered, y = TotalRevenue, fill = month_ordered)) +
  geom_col() +
  labs(
    title = "Total Sales Revenue by Month",
    x = "Month",
    y = "Total Revenue",
    fill = "Month"
  ) +
  scale_y_continuous(labels = dollar_format()) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggsave("temp_charts/plot_1.png", p1, width=10, height=6)
plots_list[[1]] <- p1

# Plot 2: Total Sales Revenue by City
city_revenue_plot_data <- df %>%
  group_by(city) %>%
  summarise(TotalRevenue = sum(total_revenue, na.rm = TRUE)) %>%
  arrange(desc(TotalRevenue))

p2 <- ggplot(city_revenue_plot_data, aes(x = reorder(city, -TotalRevenue), y = TotalRevenue, fill = city)) +
  geom_col() +
  labs(
    title = "Total Sales Revenue by City",
    x = "City",
    y = "Total Revenue",
    fill = "City"
  ) +
  scale_y_continuous(labels = dollar_format()) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggsave("temp_charts/plot_2.png", p2, width=10, height=6)
plots_list[[2]] <- p2

# Plot 3: Top 10 Most Frequently Purchased Products (for visualization clarity)
product_frequency_plot_data <- df %>%
  group_by(product) %>%
  summarise(Count = n()) %>%
  arrange(desc(Count)) %>%
  slice_head(n = 10)

p3 <- ggplot(product_frequency_plot_data, aes(x = reorder(product, -Count), y = Count, fill = product)) +
  geom_col() +
  labs(
    title = "Top 10 Most Frequently Purchased Products",
    x = "Product",
    y = "Number of Purchases",
    fill = "Product"
  ) +
  scale_y_continuous(labels = comma_format()) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggsave("temp_charts/plot_3.png", p3, width=10, height=6)
plots_list[[3]] <- p3

# Plot 4: Top 5 Products by Sales Revenue Contribution
top_5_products_revenue_plot_data <- df %>%
  group_by(product) %>%
  summarise(TotalRevenue = sum(total_revenue, na.rm = TRUE)) %>%
  arrange(desc(TotalRevenue)) %>%
  slice_head(n = 5)

p4 <- ggplot(top_5_products_revenue_plot_data, aes(x = reorder(product, -TotalRevenue), y = TotalRevenue, fill = product)) +
  geom_col() +
  labs(
    title = "Top 5 Products by Sales Revenue Contribution",
    x = "Product",
    y = "Total Revenue",
    fill = "Product"
  ) +
  scale_y_continuous(labels = dollar_format()) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggsave("temp_charts/plot_4.png", p4, width=10, height=6)
plots_list[[4]] <- p4


# 6. DASHBOARD OUTPUT
# Convert ggplot objects to plotly objects for interactivity
plotly_plots <- lapply(plots_list, ggplotly)
save_html(tagList(plotly_plots), "temp_charts/interactive_dashboard.html")