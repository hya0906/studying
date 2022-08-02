
// ChildView.h: CChildView 클래스의 인터페이스
//
#include "mmsystem.h"

#pragma once


// CChildView 창

class CChildView : public CWnd
{
// 생성입니다.
public:
	CRect clientview;
	int gap = 20;    //오선 간격
	int p_gap;       //건반두께간격
	int note = 25;   //음표 크기
	int a; //음표사이 간격 횟수
	CList <CRect, CRect&> position_list; //피아노건반위치
	void first_setting(CDC* dc, CRect clientview); //처음세팅그림
	void draw_notes(int i);//음표그리기(일단 하나 그려놓음)
	CStringArray note_name;//피아노 계명저장
	CPoint current_point;
	CList <CRect, CRect&> note_positions;//눌렀던 음의 그려질 음표위치
	//그동안 눌렀던 음의 그려질 음표 위치를 저장할 리스트 필요
	void make_sound(int i, int instruments);//음출력
	CString szSoundPath;//파일경로
	int instruments;//악기종류
	CRect removeAll_button;//지우는 버튼(사각형)영역저장
	CRect remove_button;//지우는 버튼(사각형)영역저장
	CRect next;//다음으로 넘어가는 버튼위치
	CRect prev;//그 전으로 넘어가는 버튼 위치
	int page;//악보갯수
	int current_page;//현재악보
	CChildView();

// 특성입니다.
public:

// 작업입니다.
public:

// 재정의입니다.
	protected:
	virtual BOOL PreCreateWindow(CREATESTRUCT& cs);

// 구현입니다.
public:
	virtual ~CChildView();

	// 생성된 메시지 맵 함수
protected:
	afx_msg void OnPaint();
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnLButtonDown(UINT nFlags, CPoint point);
	afx_msg void piano_sound();
	afx_msg void xylophone_sound();
};

