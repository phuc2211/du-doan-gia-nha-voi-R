#Cài đặt các thư viện cần thiết:
if (!require("xgboost")) install.packages("xgboost")
if (!require("caret")) install.packages("caret")
if (!require("corrplot")) install.packages("corrplot")

# Đọc file dữ liệu
house_price_data <- read.csv("F:/Phân tích DLL với R/house_price_new.csv")


# trực quan hóa dữ liệu
# biểu đồ cột
hist(house_price_data$price,
     col = "skyblue",
     main = "Biểu đồ phân phối giá nhà",
     xlab = "Giá nhà (đvt: 100.000 USD)",
     breaks = 30)

# ma trận tương quan
library(GGally)
ggcorr(house_price_data, geom = "blank", label = T, label_size = 3, hjust = 1, size = 3, layout.exp = 2) +
  geom_point(size = 8, aes(color = coefficient > 0, alpha = abs(coefficient) >= 0.5)) +
  scale_alpha_manual(values = c("TRUE" = 0.25, "FALSE" = 0)) +
  guides(color = F, alpha = F)# to select relevant features


#Tiền xử lý dữ liệu:
#Kiểm tra dữ liệu thiếu 
colSums(is.na(house_price_data))
#kiểm tra dữ liệu trùng lặp
sum(duplicated(house_price_data)) 

#Hiển thị dữ liệu:
library(dplyr)
glimpse(house_price_data)


# gán thuộc tính 
target_variable <- "price"  #Biến mục tiêu: biến muốn dự đoán
features <- colnames(house_price_data)[-which(colnames(house_price_data) == target_variable)]

# chia tập dữ liệu (80%/20%)
set.seed(123)  
index <- createDataPartition(house_price_data[, target_variable], p = 0.8, list = FALSE)
training_data <- house_price_data[index,]
testing_data <- rbind(house_price_data[-index,], training_data)
testing_data <- house_price_data[-index,]

# phân chia thuộc tính và biến mục tiêu
training_features <- training_data[, colnames(training_data) != target_variable]
testing_features <- testing_data[, colnames(testing_data) != target_variable]
training_target <- training_data[, target_variable]
testing_target <- testing_data[, target_variable]




library(xgboost)
library(caret)
# Chuyển đổi dữ liệu thành dạng ma trận (mô hình XGBoost chỉ nhận các kiểu dữ liệu dạng ma trận)
training_features_matrix <- as.matrix(training_features)
testing_features_matrix <- as.matrix(testing_features)

# Tạo mô hình XGBoost
model <- xgboost(data = as.matrix(training_features),
                 label = training_target,
                 nrounds = 50,  
                 eta = 0.3, 
                 max_depth = 6,          # độ sâu tối đa của cây
                 min_child_weight = 1,   # trọng số tối thiểu của một nút con
                 gamma = 0,              # tham số kiểm soát độ phức tạp của mô hình
                 colsample_bytree = 0.8, # tỷ lệ cột lấy mẫu khi xây dựng mỗi cây
                 subsample = 0.8,        # tỷ lệ dữ liệu lấy mẫu cho mỗi câye
                 objective = "reg:squarederror")  #Hàm mất mát được sử dụng là bình phương sai số R-squared


# Định nghĩa mô hình với phương thức xgboost (xgbTree)
tune_grid <- expand.grid(nrounds = 100,        # số lượng lần lặp (iterations/cây)
                         max_depth = 6,       # độ sâu tối đa của cây
                         eta = 0.3,           # tốc độ học
                         gamma = 0,           # tham số kiểm soát độ phức tạp của mô hình
                         colsample_bytree = 1, # tỷ lệ cột lấy mẫu khi xây dựng mỗi cây
                         min_child_weight = 1, # trọng số tối thiểu của một nút con
                         subsample = 1)        # tỷ lệ dữ liệu lấy mẫu cho mỗi cây

model <- train(x = training_features_matrix, 
               y = training_target, 
               method = "xgbTree", 
               trControl = trainControl(method = "cv", number = 10),  # sử dụng k-fold cross-validation
               tuneGrid = tune_grid)  # sử dụng tuneGrid để điều chỉnh số lần lặp

#print(model)

#làm dự đoán trên tập huấn luyện:
training_predictions <- predict(model, training_features_matrix)

# tính hệ số xác định R-squared
r_squared_train <- cor(training_target, training_predictions)^2
cat("R-squared của Tập huấn luyện:", r_squared_train, "\n")

# tính sai số tuyệt đối trung bình (MAE)
mae_train <- mean(abs(training_target - training_predictions))
cat("MAE của Tập huấn luyện:", mae_train, "\n")

# Tính RMSE (Root Mean Squared Error)
rmse_train <- sqrt(mean((training_target - training_predictions)^2))
cat("RMSE của Tập huấn luyện:", rmse_train, "\n")


# làm dự đoán trên tập dữ liệu testing
testing_predictions <- predict(model, testing_features_matrix)

# tính hệ số xác định R-squared 
r_squared_test <- cor(testing_target, testing_predictions)^2
cat("R-squared của Tập kiểm thử:", r_squared_test, "\n")

# tính sai số tuyệt đối trung bình (MAE)
mae_test <- mean(abs(testing_target - testing_predictions))
cat("MAE của Tập kiểm thử:", mae_test, "\n")

# Tính RMSE (Root Mean Squared Error)
rmse_test <- sqrt(mean((testing_target - testing_predictions)^2))
cat("RMSE của kiểm thử:", rmse_test, "\n")



# Hàm thu thập dữ liệu từ khách hàng 
get_user_input <- function() {
  # Hàm nhỏ để đảm bảo người dùng nhập đúng giá trị số
  read_numeric_input <- function(prompt_text) {
    repeat {
      input <- readline(prompt = prompt_text)
      numeric_input <- suppressWarnings(as.numeric(input))
      if (!is.na(numeric_input)) {
        return(numeric_input)  # Trả về giá trị số hợp lệ
      } else {
        cat("Giá trị không hợp lệ. Vui lòng nhập lại.\n")
      }
    }
  }
  
  # Nhập các giá trị từ người dùng
  MedInc <- read_numeric_input("Nhập thu nhập trung bình năm của bạn (đvt: 10000 USD) (vd: 8.33): ")
  HouseAge <- read_numeric_input("Nhập tuổi của căn nhà (vd: 12): ")
  AveRooms <- read_numeric_input("Nhập số phòng bạn muốn có trong nhà (vd: 5): ")
  Population <- read_numeric_input("Nhập số hộ dân bạn muốn xung quanh (vd: 500): ")
  AveBedrms <- read_numeric_input("Nhập số phòng ngủ bạn muốn (vd: 4): ")
  AveOccup <- read_numeric_input("Nhập số người trong nhà (vd: 5): ")
  
  # Các giá trị mặc định
  Latitude <- 35.631861
  Longitude <- -119.569704
  
  # Kết hợp tất cả các giá trị thành một vector
  features <- c(MedInc, HouseAge, AveRooms, Population, AveBedrms, AveOccup, Latitude, Longitude)
  
  # Chuyển đổi thành ma trận 1 dòng
  matrix(features, nrow = 1)
}

# Hàm dự đoán giá nhà
predict_house_price <- function(model, user_input) {
  prediction <- predict(model, user_input)
  return(prediction)
  }

# Gọi lệnh nhập dữ liệu
user_input <- get_user_input()

#dự đoán giá nhà
predicted_price <- predict_house_price(model, user_input)
cat("Dự đoán giá nhà là:", round(predicted_price, 3)*100000 ,"USD", "\n")







