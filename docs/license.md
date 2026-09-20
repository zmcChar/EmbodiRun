# License and relicensing

EmbodiRun is distributed under the **Apache License, Version 2.0**. See
[`LICENSE`](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/LICENSE),
[`NOTICE`](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/NOTICE), and
[`THIRD_PARTY_NOTICES.md`](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/THIRD_PARTY_NOTICES.md).

## Why we moved from MIT to Apache-2.0

Both repositories were previously MIT. We changed to Apache-2.0 before the
public release for five concrete reasons:

1. **An explicit patent grant.** Apache-2.0 section 3 grants every user a
   patent license covering the contributor's contributions, and terminates that
   grant if the user sues over the work. MIT is silent on patents. For a
   runtime and inference engine that touches GPU kernels, graph capture, and
   communication scheduling, this protection matters to users and to us.
2. **Trademark clarity.** Apache-2.0 section 6 does **not** grant rights to the
   licensor's trade names or trademarks. That lets the project keep the
   "EmbodiRun" and "EmbodiInfer" names meaningful, while MIT leaves naming
   entirely unaddressed.
3. **Clear contribution terms.** Apache-2.0 section 5 states that any
   contribution intentionally submitted for inclusion is licensed under the
   same terms unless stated otherwise. This removes the "which license is this
   patch under?" ambiguity that MIT-only projects often hit when external
   contributors join.
4. **Consistency with the code we already ship.** The inference engine vendors
   and adapts several Apache-2.0 leaves (OpenVLA-OFT / Prismatic, NVIDIA
   Isaac-GR00T, Cosmos-Policy / Wan2.1 VAE, diffusers conventions). Using the
   same license at the project level makes attribution and redistribution
   simpler and less error-prone; see `THIRD_PARTY_NOTICES.md`.
5. **Ecosystem expectations for infrastructure.** Apache-2.0 is the common
   license for deployment/runtime infrastructure (Kubernetes, PyTorch, and most
   of the surrounding stack). It is familiar to the companies and labs that
   might embed EmbodiRun in a product, which lowers adoption friction.

## What does not change

- **Still permissive.** Commercial use, modification, redistribution, and
  private use remain allowed. There is no copyleft obligation on your own code.
- **Third-party licenses are unchanged.** Robot SDKs, simulators, models, and
  datasets keep their own terms; the project license does not relicense them.
- **No product or API change.** The HTTP/WirelessComm protocol, packaging, and
  runtime behavior are unaffected by the license.

## Compatibility notes

- Apache-2.0 combines cleanly with our MIT, BSD, and Apache-2.0 dependencies.
- Apache-2.0 is not compatible with GPLv2-only code in a single combined work.
  We do not vendor GPLv2-only code. The optional `paramiko` dependency is
  LGPL-2.1 and is used as a separate, dynamically imported library.
- `Isaac Sim` is a proprietary NVIDIA dependency selected through an optional
  install group; it is not distributed here and its EULA is the user's
  responsibility.

## Contributor consent

Relicensing from MIT to Apache-2.0 requires the agreement of the copyright
holders whose contributions are already in the repository. Until that record is
complete, the previous MIT terms remain available from git history.

Contributors with commits in **EmbodiRun** include:

- yufoo1 `<yufoo1.cs@gmail.com>`
- cc `<cclonelycc@outlook.com>`
- hootandy321 / Xingyu Liu `<133196559+hootandy321@users.noreply.github.com>`
- Ao Zhou `<425109310@qq.com>`
- 刘兴宇 `<hootandy@outlook.com>`

Contributors with commits in **EmbodiInfer** include:

- yufoo1 `<yufoo1.cs@gmail.com>`
- Longxmas `<1185267696@qq.com>`
- Liu xingyu `<133196559+hootandy321@users.noreply.github.com>`
- Ao Zhou `<425109310@qq.com>`
- cc `<cclonelycc@outlook.com>`

If you are listed above and agree to the change, record your consent on the
relicensing tracking issue (or in the pull request that introduces this file)
with a comment or a `Signed-off-by:` line. If you do not agree, say so there so
the maintainers can resolve it before publication.

## 中文摘要

仓库此前使用 MIT，现已改为 **Apache-2.0**，原因：① 显式专利授权与专利终止条款，MIT 没有；② 明确不授予商标权，便于保护 EmbodiRun / EmbodiInfer 名称；③ 有明确的贡献条款，避免外部贡献的许可歧义；④ 仓库内 vendored 代码本身是 Apache-2.0，项目级统一更省事；⑤ 部署/运行时类基础设施普遍采用 Apache-2.0，商业采用更顺畅。

改动不影响：仍是宽松许可、可商用；第三方组件各自保留许可证；协议与运行时行为不变。

从 MIT 改为 Apache-2.0 需要现有版权持有人同意。请上表列出的贡献者在 relicensing tracking issue 中留言或使用 `Signed-off-by:` 表示同意；不同意也请说明，维护者会在公开发布前处理。
