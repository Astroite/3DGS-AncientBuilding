"""离屏检查 P1 GUI 壳：面板内容必须来自 services，证据存在不等于阶段成功。

不启动真实重建、训练或 SDK 验收；依赖检查只验证按钮与工作线程的接线，不自动运行。
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from tests.factories import DIGEST, RUN_ID, STRAY_ID, build_tree  # noqa: E402
from gsstudio.interfaces.desktop import theme  # noqa: E402
from gsstudio.interfaces.desktop.app import MainWindow, shorten_path  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _tree_rows(group) -> list[str]:
    return [group.child(index).text(0) for index in range(group.childCount())]


def test_theme_follows_the_visual_spec(qapp):
    stylesheet = theme.build_stylesheet()
    assert theme.APP_BG == "#121719"
    assert theme.ACCENT == "#77D6C3"
    assert "#77D6C3" in stylesheet
    assert "Segoe UI" in stylesheet
    assert shorten_path("D:\\Project\\3DGS\\Data\\loc\\scene") == "D:\\Project\\…\\loc\\scene"


def test_project_tree_reports_status_and_problems_from_manifests(tmp_path, qapp):
    built = build_tree(tmp_path)
    root = built["root"]
    (root / "loc" / "scene" / STRAY_ID / "inputs").mkdir(parents=True)
    window = MainWindow(root)

    assert window.tree.topLevelItemCount() == 1
    location = window.tree.topLevelItem(0)
    assert location.text(0) == "loc"
    assert "草稿" in location.text(1)

    window.tree.expandItem(location)
    scene = location.child(0)
    assert scene.text(0) == "scene"

    window.tree.expandItem(scene)
    capture_group, run_group = scene.child(0), scene.child(1)
    assert capture_group.text(0) == "Capture（1）"
    assert _tree_rows(capture_group) == ["capture"]
    assert sorted(_tree_rows(run_group)) == sorted([RUN_ID, STRAY_ID])
    stray = next(run_group.child(i) for i in range(run_group.childCount()) if run_group.child(i).text(0) == STRAY_ID)
    assert "不可读" in stray.text(1)


def test_run_detail_shows_stages_and_evidence_without_inventing_success(tmp_path, qapp):
    built = build_tree(tmp_path)
    window = MainWindow(built["root"])
    location = window.tree.topLevelItem(0)
    window.tree.expandItem(location)
    scene = location.child(0)
    window.tree.expandItem(scene)
    run_group = scene.child(1)
    window.tree.setCurrentItem(run_group.child(0))

    assert window.center.currentWidget() is window.run_panel
    assert window.run_panel.title.text() == f"Run {RUN_ID}"
    assert window.run_panel.stages.rowCount() == 4
    assert [window.run_panel.stages.item(row, 0).text() for row in range(4)] == [
        "preprocess",
        "mask",
        "reconstruct",
        "qa",
    ]
    assert all(window.run_panel.stages.item(row, 1).text() == "待执行" for row in range(4))

    evidence = {
        window.run_panel.evidence.item(row, 0).text(): window.run_panel.evidence.item(row, 1).text()
        for row in range(window.run_panel.evidence.rowCount())
    }
    assert evidence["prepared-input"] == "✓ 存在"
    assert evidence["qa-report"] == "! 缺失"
    # A present artifact documents work; the stage records stay untouched.
    assert all(window.run_panel.stages.item(row, 1).text() == "待执行" for row in range(4))

    config = {
        window.run_panel.config.item(row, 0).text(): window.run_panel.config.item(row, 1).text()
        for row in range(window.run_panel.config.rowCount())
    }
    assert config["source_kind"] == "equirect_video"
    assert config["mask_discard_threshold"] == "0.005"


def test_missing_prepared_input_is_listed_as_a_locatable_problem(tmp_path, qapp):
    built = build_tree(tmp_path)
    root = built["root"]
    (root / "loc" / "scene" / RUN_ID / "inputs" / "primary" / DIGEST / "dataset.json").unlink()
    window = MainWindow(built["root"])
    location = window.tree.topLevelItem(0)
    window.tree.expandItem(location)
    scene = location.child(0)
    window.tree.expandItem(scene)
    window.tree.setCurrentItem(scene.child(1).child(0))

    assert window.inspector.problems.count() == 1
    text = window.inspector.problems.item(0).text()
    assert "missing_input" in text
    assert window.inspector.problems.item(0).toolTip()


def test_scene_selection_lists_captures_and_runs(tmp_path, qapp):
    built = build_tree(tmp_path)
    window = MainWindow(built["root"])
    location = window.tree.topLevelItem(0)
    window.tree.expandItem(location)
    scene = location.child(0)
    window.tree.setCurrentItem(scene)

    assert window.center.currentWidget() is window.scene_panel
    assert window.scene_panel.captures.rowCount() == 1
    assert window.scene_panel.captures.item(0, 0).text() == "capture"
    assert window.scene_panel.runs.rowCount() == 1
    assert window.scene_panel.runs.item(0, 0).text() == RUN_ID
