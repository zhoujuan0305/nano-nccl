# Tofino 单口双机 L2 验证

`enable_tofino_static_l2.py` 用于已运行的 Tofino1 `l2learn` pipeline。默认只打印 dry-run 计划。加 `--apply --configure-ports` 后，它先读当前 `$PORT`、`forward` 和 `proxy_arp` 状态；只对指定的两个 dev_port 添加 100G/FEC/lanes 配置并启用，然后写入两条 proxy ARP 和两条精确转发项。已有配置不匹配或转发表精确 key 冲突时退出，不会覆盖已有条目，也不改其他端口。

## 启动前检查

在目标交换机上以 root 执行。只要发现 switchd、内嵌 ASIC controller、BFRT/status listener 或已有持久会话，就停止启动并先核实 owner。

```bash
hostname
id
date -Is
pgrep -af '[b]f_switchd|[b]frt|[t]ofino|[p]4runtime' || true
ss -ltnp | grep -E 'State|:(50052|9999|7777)[[:space:]]' || true
tmux list-sessions 2>/dev/null || true
```

## 启动 `l2learn`

使用目标交换机上已安装的 Tofino1 SDE 与 `l2learn` 编译产物。将下列占位路径换成本机 SDE 和本次日志目录；不要覆盖既有 session。

```bash
SDE_ROOT=/path/to/tofino-sde
RUN_ID=unique-run-id
SESSION="nano-nccl-gdr-${RUN_ID}"
LOG_DIR="/path/to/task-logs/${RUN_ID}"
mkdir -p "$LOG_DIR"

tmux new-session -d -s "$SESSION" \
  "cd '$SDE_ROOT' && exec ./run_switchd.sh -p l2learn >>'$LOG_DIR/switch.log' 2>&1"
```

验收 `bf_switchd` 命令使用 `l2learn.conf`，启动日志报告 `dev_id 0 initialized`，BFRT `:50052`、status `:7777`（若本机配置启用则 `:9999`）监听，tmux session 仍存在。启动 `l2learn` 会重新初始化 ASIC；只在上面的 owner 检查确认空闲并获得维护者对网络窗口的授权后执行。

## 配置两个端点

先从当前主机接口和 lab inventory 取 dev_port、IPv4、MAC、speed、FEC。不要把现场 NIC 地址写入此仓库。把 helper 放到交换机本次日志目录，使用随该 SDE 提供的 Python 运行时。

```bash
PYTHONPATH="$SDE_ROOT/install/lib/python3.5/site-packages/tofino:$SDE_ROOT/install/lib/python3.5/site-packages/tofino/bfrt_grpc" \
  python3.5 enable_tofino_static_l2.py \
  --dev-port-a "$DEV_PORT_A" --ip-a "$IP_A" --mac-a "$MAC_A" \
  --dev-port-b "$DEV_PORT_B" --ip-b "$IP_B" --mac-b "$MAC_B" \
  --speed 100G --fec NONE --dry-run
```

检查 dry-run 输出后，显式添加 `--configure-ports --apply` 执行。程序要求两端 IPv4 属于同一 `/24`，端口配置与预期 speed/FEC 一致；它只添加缺少的端口项，只在两端 `$PORT_ENABLE`/`$PORT_UP` 通过后写 ARP 和 L2 表，并对所有写入做读回校验。

最后在两台主机分别绑定这条数据口测试对端：

```bash
ping -I "$IFACE_A" -c 3 "$IP_B"
ping -I "$IFACE_B" -c 3 "$IP_A"
```

两向都应收到回应。ping 通过证明这条 IP/L2 路径可用；RDMA/GDR 仍需由项目的两机 smoke 验证。

## 恢复

完成 RDMA/GDR 验证后，若需回到本次启动前的状态，先确认网络不再被使用，再核验以下 session 确实由本次任务启动：

```bash
tmux display-message -t "$SESSION" -p '#{session_name}'
pgrep -af '[b]f_switchd'
```

确认 session 名和 `bf_switchd` 完整命令匹配本次记录后，只关闭该 session：

```bash
tmux kill-session -t "$SESSION"
```

不要用 `pkill`、`killall` 或停止其他 session。确认该 PID 已退出、`:50052`/`:7777` 不再监听且本次 session 消失。停止本次 `bf_switchd` 会移除由该 pipeline 建立的两个端口配置、两条转发项和两条 ARP 项；若之后有其他 owner 启动了服务，不要执行这一步。
