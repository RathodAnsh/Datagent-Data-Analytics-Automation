
    options(warn=-1)
    suppressPackageStartupMessages(library(tidyverse))
    suppressPackageStartupMessages(library(jsonlite))
    suppressPackageStartupMessages(library(lubridate))
    suppressPackageStartupMessages(library(readr))

    # 1. READ DATA (Safe Read)
    safe_path <- 'C:/Users/admin/Desktop/Data Analyst Agent/temp_data/raw_upload.csv'
    df <- tryCatch({
      read_csv(safe_path, show_col_types = FALSE, guess_max = 10000)
    }, error = function(e) {
      read.csv(safe_path, stringsAsFactors = FALSE)
    })

    # 2. CLEAN COLUMN NAMES
    names(df) <- tolower(gsub("[^a-zA-Z0-9_]+", "_", names(df)))
    names(df) <- gsub("^_+|_+$", "", names(df))

    # 3. INTELLIGENT TYPE CONVERSION
    clean_numeric <- function(x) {
      clean_str <- gsub("[$,€£]", "", x)
      clean_str <- gsub(",", "", clean_str)
      clean_str <- gsub("\\((.*)\\)", "-\\1", clean_str)
      return(as.numeric(clean_str))
    }

    for(col in names(df)) {
      if(is.character(df[[col]])) {
        non_na_vals <- df[[col]][!is.na(df[[col]])]
        if(length(non_na_vals) > 0) {
          matches_num <- grep("^[\\$€£,0-9\\.\\-]+$", non_na_vals)
          if(length(matches_num) / length(non_na_vals) > 0.7) {
             df[[col]] <- clean_numeric(df[[col]])
          } else {
             parsed_dates <- parse_date_time(non_na_vals, orders = c("mdy", "dmy", "ymd", "HMS", "ymd HMS"), quiet = TRUE)
             if(sum(!is.na(parsed_dates)) / length(non_na_vals) > 0.7) {
               df[[col]] <- parse_date_time(df[[col]], orders = c("mdy", "dmy", "ymd", "HMS", "ymd HMS"), quiet = TRUE)
             }
          }
        }
      }
    }

    # 4. REMOVE DUPLICATES
    initial_rows <- nrow(df)
    df <- df %>% distinct()
    duplicates_removed <- initial_rows - nrow(df)

    # 5. GENERATE PROFILE JSON
    stats_list <- list()
    numeric_cols <- c()
    date_cols <- c()
    categorical_cols <- c()

    for(col in names(df)) {
      col_type <- class(df[[col]])[1]
      missing_count <- sum(is.na(df[[col]]))
      unique_count <- n_distinct(df[[col]])
      
      col_stat <- list(
        dtype = col_type,
        missing = missing_count,
        unique = unique_count
      )

      if(is.numeric(df[[col]])) {
        numeric_cols <- c(numeric_cols, col)
        col_stat$min <- min(df[[col]], na.rm=TRUE)
        col_stat$max <- max(df[[col]], na.rm=TRUE)
        col_stat$mean <- mean(df[[col]], na.rm=TRUE)
      } else if (inherits(df[[col]], "POSIXt") || inherits(df[[col]], "Date")) {
        date_cols <- c(date_cols, col)
        df[[col]] <- as.character(df[[col]]) 
      } else {
        categorical_cols <- c(categorical_cols, col)
      }
      
      stats_list[[col]] <- col_stat
    }

    profile_data <- list(
      timestamp = format(Sys.time(), "%Y-%m-%dT%H:%M:%S"),
      dataset_shape = list(rows = nrow(df), columns = ncol(df)),
      columns_list = names(df),
      numeric_cols = numeric_cols,
      date_cols = date_cols,
      categorical_cols = categorical_cols,
      sample_data = head(df, 5),
      statistics = stats_list,
      duplicates_removed = duplicates_removed
    )

    # 6. SAVE OUTPUTS (Base R Write)
    write.csv(df, "C:/Users/admin/Desktop/Data Analyst Agent/temp_data/source.csv", row.names = FALSE)
    write_json(profile_data, "C:/Users/admin/Desktop/Data Analyst Agent/data_profiles/data_profiling.json", auto_unbox = TRUE, pretty = TRUE)
    