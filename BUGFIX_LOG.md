# Bug修复日志

## 2025-06-23: 数据库连接认证问题修复

### 问题描述
- **错误信息**: `'cryptography' package is required for sha256_password or caching_sha2_password auth methods`
- **影响范围**: 数据更新器无法连接MySQL数据库
- **触发时间**: 2025-06-23 20:09

### 问题分析
1. **根本原因**: MySQL连接认证方式版本兼容性问题
2. **具体原因**: `cryptography` 包版本过新 (45.0.4)，与 PyOpenSSL 等依赖库冲突
3. **排除原因**: 不是代码修改导致，最近提交只涉及图片管理重构

### 解决方案
1. **降级cryptography包**:
   ```bash
   pip install "cryptography>=41.0.5,<42"
   ```
   - 从 45.0.4 → 41.0.7
   - 解决与 PyOpenSSL 版本冲突

2. **升级PyMySQL驱动**:
   ```bash
   pip install PyMySQL --upgrade
   ```

### 验证结果
- ✅ 数据库基础连接测试通过
- ✅ 数据更新器正常运行
- ✅ 成功更新500+只股票数据
- ✅ 平均每只股票更新2条新数据

### 预防措施
- 建议添加 requirements.txt 固定关键依赖版本
- 定期检查依赖包兼容性

### 修复人员
- 系统维护

### 状态
- [x] 已修复
- [x] 已验证
- [x] 已记录 