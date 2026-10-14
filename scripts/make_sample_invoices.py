"""生成合成发票样本（PNG）+ ground truth JSON 标注。

用法：uv run python scripts/make_sample_invoices.py
输出：samples/*.png 与 samples/*.labels.json（标注文件即评测集）
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = PROJECT_ROOT / "samples"

# macOS 自带中文字体
FONT_CANDIDATES = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
]


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    raise RuntimeError("未找到系统中文字体，请安装 PingFang/宋体等")


DIGITS = "零壹贰叁肆伍陆柒捌玖"
UNITS = ["", "拾", "佰", "仟"]
BIG_UNITS = ["", "万", "亿"]


def to_cn_amount(amount: float) -> str:
    """把金额转为人民币大写（整数部分到元，角分忽略大写化细节）。"""
    yuan = int(round(amount))
    if yuan == 0:
        return "零元整"
    parts = []
    section_idx = 0
    while yuan > 0:
        section = yuan % 10000
        if section:
            text = ""
            zero_pending = False
            for pos in range(3, -1, -1):
                digit = (section // 10**pos) % 10
                if digit:
                    if zero_pending:
                        text += "零"
                        zero_pending = False
                    text += DIGITS[digit] + UNITS[pos]
                elif text:
                    zero_pending = True
            parts.append(text + BIG_UNITS[section_idx])
        else:
            parts.append("")
        yuan //= 10000
        section_idx += 1
    return "".join(reversed(parts)).strip("零") + "元整"


def draw_invoice(spec: dict, blurred: bool = False) -> Image.Image:
    """按增值税发票常见版式绘制一张发票。"""
    W, H = 1000, 620
    img = Image.new("RGB", (W, H), "#fdfdf8")
    d = ImageDraw.Draw(img)
    f_title = load_font(34)
    f_head = load_font(22)
    f_body = load_font(20)
    f_small = load_font(16)

    d.rectangle([20, 20, W - 20, H - 20], outline="#d48b3c", width=3)
    d.text((W // 2, 55), spec["title"], font=f_title, fill="#333", anchor="mm")
    d.text((W // 2, 95), "发票联", font=f_head, fill="#666", anchor="mm")

    # 发票代码 / 号码 / 日期
    d.text((60, 130), f"发票代码：{spec.get('invoice_code', '—')}", font=f_body, fill="#333")
    d.text((400, 130), f"发票号码：{spec['invoice_number']}", font=f_body, fill="#c0392b")
    d.text((720, 130), f"开票日期：{spec['invoice_date']}", font=f_body, fill="#333")

    # 购销双方
    d.line([40, 170, W - 40, 170], fill="#d48b3c", width=2)
    d.text((60, 185), f"购买方名称：{spec['buyer_name']}", font=f_body, fill="#333")
    d.text((60, 215), f"购买方税号：{spec.get('buyer_tax_id') or '—'}", font=f_body, fill="#333")
    d.text((60, 250), f"销售方名称：{spec['seller_name']}", font=f_body, fill="#333")
    d.text((60, 280), f"销售方税号：{spec.get('seller_tax_id') or '—'}", font=f_body, fill="#333")

    # 明细表
    table_top = 320
    d.line([40, table_top, W - 40, table_top], fill="#d48b3c", width=2)
    headers = ["货物或应税劳务、服务名称", "规格型号", "数量", "单价", "金额", "税率"]
    xs = [50, 380, 540, 640, 760, 880]
    for x, h in zip(xs, headers, strict=True):
        d.text((x, table_top + 10), h, font=f_small, fill="#333")
    d.line([40, table_top + 40, W - 40, table_top + 40], fill="#d48b3c", width=1)
    y = table_top + 50
    for item in spec["items"]:
        d.text((xs[0], y), item["name"], font=f_small, fill="#333")
        d.text((xs[1], y), item.get("specification", "—"), font=f_small, fill="#333")
        d.text((xs[2], y), str(item.get("quantity", "—")), font=f_small, fill="#333")
        d.text((xs[3], y), str(item.get("unit_price", "—")), font=f_small, fill="#333")
        d.text((xs[4], y), f"{item['amount']:.2f}", font=f_small, fill="#333")
        d.text((xs[5], y), item.get("tax_rate", "—"), font=f_small, fill="#333")
        y += 30

    # 金额区
    y_amount = max(y + 20, 480)
    d.text((60, y_amount), f"合  计：¥{spec['amount_without_tax']:.2f}", font=f_body, fill="#333")
    d.text((60, y_amount + 30), f"税  额：¥{spec['tax_amount']:.2f}", font=f_body, fill="#333")
    d.text((480, y_amount + 10), f"价税合计（大写）：{to_cn_amount(spec['total_amount'])}", font=f_small, fill="#333")
    d.text(
        (480, y_amount + 40),
        f"（小写）¥{spec['total_amount']:.2f}",
        font=f_head,
        fill="#c0392b",
    )

    # 备注 + 章
    if spec.get("remarks"):
        d.text((60, y_amount + 75), f"备注：{spec['remarks']}", font=f_small, fill="#555")
    d.ellipse([760, 500, 940, 660], outline="#c0392b", width=3)
    d.text((850, 560), spec["seller_name"], font=f_small, fill="#c0392b", anchor="mm")
    d.text((850, 585), "发票专用章", font=f_small, fill="#c0392b", anchor="mm")

    if blurred:
        img = img.filter(ImageFilter.GaussianBlur(radius=2.5))
        img = img.rotate(random.uniform(-2.5, 2.5), expand=True, fillcolor="#fdfdf8")
    return img


def build_specs(config: dict) -> list[dict]:
    company = config["company"]
    comp_name, comp_tax = company["name"], company["tax_id"]
    seller = "杭州某某商贸有限公司"
    seller_tax = "91330100MA28YYYYYY"

    specs = [
        dict(
            sample_id="01_normal_special",
            title="浙江增值税专用发票",
            invoice_code="044032000111",
            invoice_number="28374651",
            invoice_date="2026年09月30日",
            buyer_name=comp_name,
            buyer_tax_id=comp_tax,
            seller_name=seller,
            seller_tax_id=seller_tax,
            amount_without_tax=1000.0,
            tax_amount=130.0,
            total_amount=1130.0,
            items=[dict(name="办公用纸A4", specification="70g 500张/包", quantity=20, unit_price=50.0, amount=1000.0, tax_rate="13%")],
            remarks="",
        ),
        dict(
            sample_id="02_normal_personal",
            title="浙江增值税普通发票",
            invoice_code="044032000112",
            invoice_number="31928374",
            invoice_date="2026年10月08日",
            buyer_name="个人",
            buyer_tax_id=None,
            seller_name="杭州某某餐饮管理有限公司",
            seller_tax_id="91330100MA29ZZZZZZ",
            amount_without_tax=380.0,
            tax_amount=0.0,
            total_amount=380.0,
            items=[dict(name="餐饮服务*餐费", specification="—", quantity=1, unit_price=380.0, amount=380.0, tax_rate="免税")],
            remarks="",
        ),
        dict(
            sample_id="03_digital",
            title="电子发票（增值税专用发票）",
            invoice_code=None,
            invoice_number="25612000000123456789",
            invoice_date="2026年10月10日",
            buyer_name=comp_name,
            buyer_tax_id=comp_tax,
            seller_name=seller,
            seller_tax_id=seller_tax,
            amount_without_tax=2000.0,
            tax_amount=260.0,
            total_amount=2260.0,
            items=[dict(name="打印耗材硒鼓", specification="HP-88A", quantity=4, unit_price=500.0, amount=2000.0, tax_rate="13%")],
            remarks="",
        ),
        dict(
            sample_id="04_wrong_taxid",
            title="浙江增值税专用发票",
            invoice_code="044032000113",
            invoice_number="41526374",
            invoice_date="2026年10月11日",
            buyer_name=comp_name,
            buyer_tax_id="91330100MA00WRONGX",  # 税号错误
            seller_name=seller,
            seller_tax_id=seller_tax,
            amount_without_tax=800.0,
            tax_amount=104.0,
            total_amount=904.0,
            items=[dict(name="电脑配件", specification="—", quantity=2, unit_price=400.0, amount=800.0, tax_rate="13%")],
            remarks="",
        ),
        dict(
            sample_id="05_sensitive_giftcard",
            title="浙江增值税普通发票",
            invoice_code="044032000114",
            invoice_number="52637485",
            invoice_date="2026年10月12日",
            buyer_name=comp_name,
            buyer_tax_id=comp_tax,
            seller_name="杭州某某百货有限公司",
            seller_tax_id="91330100MA30AAAAAA",
            amount_without_tax=3000.0,
            tax_amount=0.0,
            total_amount=3000.0,
            items=[dict(name="购物卡", specification="面值1000元", quantity=3, unit_price=1000.0, amount=3000.0, tax_rate="不征税")],
            remarks="",
        ),
        dict(
            sample_id="06_serial_with_01",
            title="浙江增值税专用发票",
            invoice_code="044032000111",
            invoice_number="28374655",  # 与样本1连号（差 4 ≤ 5）
            invoice_date="2026年09月30日",
            buyer_name=comp_name,
            buyer_tax_id=comp_tax,
            seller_name=seller,
            seller_tax_id=seller_tax,
            amount_without_tax=500.0,
            tax_amount=65.0,
            total_amount=565.0,
            items=[dict(name="签字笔", specification="0.5mm黑色", quantity=50, unit_price=10.0, amount=500.0, tax_rate="13%")],
            remarks="",
        ),
        dict(
            sample_id="07_fake_invoice",
            title="浙江增值税专用发票",
            invoice_code="044032000999",
            invoice_number="99999999",  # 预置进 Mock 注册表的假票
            invoice_date="2026年10月13日",
            buyer_name=comp_name,
            buyer_tax_id=comp_tax,
            seller_name=seller,
            seller_tax_id=seller_tax,
            amount_without_tax=5000.0,
            tax_amount=650.0,
            total_amount=5650.0,
            items=[dict(name="咨询服务费", specification="—", quantity=1, unit_price=5000.0, amount=5000.0, tax_rate="13%")],
            remarks="",
        ),
        dict(
            sample_id="08_blurred",
            title="浙江增值税普通发票",
            invoice_code="044032000115",
            invoice_number="63748596",
            invoice_date="2026年09月15日",
            buyer_name=comp_name,
            buyer_tax_id=comp_tax,
            seller_name="杭州某某出租车有限公司",
            seller_tax_id="91330100MA31BBBBBB",
            amount_without_tax=200.0,
            tax_amount=0.0,
            total_amount=200.0,
            items=[dict(name="客运服务费", specification="—", quantity=1, unit_price=200.0, amount=200.0, tax_rate="免税")],
            remarks="",
        ),
    ]
    return specs


def to_label(spec: dict) -> dict:
    """ground truth 标注（与 InvoiceData 字段对齐）。"""
    return dict(
        sample_id=spec["sample_id"],
        invoice_type=spec["title"],
        invoice_code=spec.get("invoice_code"),
        invoice_number=spec["invoice_number"],
        invoice_date=spec["invoice_date"],
        buyer_name=spec["buyer_name"],
        buyer_tax_id=spec.get("buyer_tax_id"),
        seller_name=spec["seller_name"],
        seller_tax_id=spec.get("seller_tax_id"),
        amount_without_tax=spec["amount_without_tax"],
        tax_amount=spec["tax_amount"],
        total_amount=spec["total_amount"],
        items=[i["name"] for i in spec["items"]],
        blurred=spec["sample_id"] == "08_blurred",
    )


def main() -> None:
    import yaml

    config = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    SAMPLES_DIR.mkdir(exist_ok=True)
    specs = build_specs(config)
    labels = []
    for spec in specs:
        img = draw_invoice(spec, blurred=spec["sample_id"] == "08_blurred")
        out = SAMPLES_DIR / f"{spec['sample_id']}.png"
        img.save(out)
        labels.append(to_label(spec))
        print(f"生成 {out.name}")
    (SAMPLES_DIR / "labels.json").write_text(
        json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"共 {len(labels)} 张，标注写入 samples/labels.json")


if __name__ == "__main__":
    sys.exit(main())
