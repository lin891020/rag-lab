"""Each chain against values read by hand from the PDF pages (and the clean tables)."""

from rules import (
    ampacity,
    corrected_ampacity,
    dwelling_other_demand,
    ground_conductor,
    lighting_demand,
    motor_branch_conductor,
    receptacle_demand,
    stranded_ladder,
)


def test_c1_ampacity_cells_and_blanks():
    a = ampacity(conduit="PVC管", insulation=60, kind="絞線", size="22", conductors=3)
    assert (a.kind, a.value, a.unit) == ("number", 60.0, "A") and a.gold_rows == [
        ("表二五～五", "絞線 22")
    ]
    assert (
        ampacity(conduit="金屬管", insulation=75, kind="單線", size="2.0", conductors=4).value
        == 21.0
    )
    assert (
        ampacity(conduit="金屬管", insulation=60, kind="絞線", size="3.5", conductors=9).value
        == 13.0
    )
    blank = ampacity(conduit="PVC管", insulation=60, kind="絞線", size="400", conductors=4)
    assert blank.kind == "abstain" and "空白" in blank.reason
    assert (
        ampacity(conduit="PVC管", insulation=60, kind="絞線", size="24", conductors=3).kind
        == "abstain"
    )


def test_c2_temperature_and_count_corrections():
    a = corrected_ampacity(
        conduit="PVC管", insulation=60, kind="絞線", size="22", conductors=3, ambient=45
    )
    assert a.value == 46.2 and a.inputs == {"ampacity": 60.0, "temp_factor": 0.77}
    assert ("表二五～七", "41 - 45") in a.gold_rows
    b = corrected_ampacity(
        conduit="金屬管", insulation=90, kind="絞線", size="8", conductors=4, ambient=8
    )
    assert b.inputs["temp_factor"] == 1.21
    c = corrected_ampacity(conduit="PVC管", insulation=60, kind="絞線", size="22", conductors=12)
    assert c.value == 30.0 and c.inputs == {"ampacity": 60.0, "count_factor": 0.5}
    assert ("表二五～六", "10 - 20") in c.gold_rows
    d = corrected_ampacity(
        conduit="PVC管", insulation=60, kind="絞線", size="22", conductors=3, ambient=58
    )
    assert d.kind == "abstain" and "空白" in d.reason  # 60℃ column ends at 51-55


def test_c3_motor_branch_conductor_on_the_pvc_ladder():
    a = motor_branch_conductor(hp=15, volts=380)
    assert a.kind == "ladder" and a.value == 8.0 and a.unit == "mm²"
    assert a.inputs == {"flc": 23.2, "factor": 1.25, "required": 29.0, "ampacity": 33.0}
    assert a.gold_rows == [("表二五八～三", "15"), ("表二五～五", "絞線 8")]
    assert a.ladder == stranded_ladder("表二五～五") and a.ladder[:3] == [3.5, 5.5, 8.0]
    assert (
        motor_branch_conductor(hp=50, volts=380).value == 50.0
    )  # 73.5 × 1.25 = 91.9 -> 50 mm² (100 A)
    assert motor_branch_conductor(hp=1, volts=380).value == 3.5  # 2.5 A but never below 3.5 mm²
    assert motor_branch_conductor(hp=3, volts=115).kind == "abstain"  # "－"
    assert motor_branch_conductor(hp=4, volts=380).kind == "abstain"  # no 4 hp row
    assert (
        motor_branch_conductor(hp=10, volts=380, conductors=6).value == 8.0
    )  # 20.25 A; 8 mm² in 5-6 col = 25


def test_c4_demand_factors_are_tiered():
    a = lighting_demand(place="住宅", watts=50000)
    assert a.value == 19450.0 and a.inputs["pct2"] == 0.35
    assert lighting_demand(place="醫院註", watts=60000).value == 22000.0
    assert (
        lighting_demand(
            place="飯店、旅館及汽車旅館，包括不提供房客烹飪用電器具之公寓式房屋註", watts=150000
        ).value
        == 57000.0
    )
    assert lighting_demand(place="大賣場(倉儲)", watts=20000).value == 16250.0
    assert lighting_demand(place="其他", watts=8000).value == 8000.0
    assert (
        lighting_demand(place="學校", watts=8000).value == 8000.0
    )  # any other place: the 其他 row, 100 %
    assert receptacle_demand(kva=30).value == 20.0 and receptacle_demand(kva=6).value == 6.0
    assert dwelling_other_demand(kw=25).value == 16.0


def test_c5_ground_conductor_range_rows():
    a = ground_conductor(ocpd_a=75)
    assert (
        a.value == 8.0
        and a.gold_rows == [("表九三～二", "100以下")]
        and a.ladder[:4] == [2.0, 3.5, 5.5, 8.0]
    )
    assert ground_conductor(ocpd_a=20).value == 2.0
    solid = ground_conductor(ocpd_a=20, kind="單線")
    assert solid.value == 1.6 and solid.ladder == [1.6, 2.0]  # the solid column's own ladder
    assert ground_conductor(ocpd_a=75, kind="單線").kind == "abstain"
    assert ground_conductor(ocpd_a=7000).kind == "abstain"
