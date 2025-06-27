# 第一阶段: 构建阶段
FROM node:18.18.2-alpine3.18 AS builder

WORKDIR /workspace/app

# 复制所有文件以进行构建
COPY . .
RUN npm install

EXPOSE 41242 41242

# 使用进程管理器启动应用 (推荐生产环境)
CMD ["npm", "run", "agents:coder"]