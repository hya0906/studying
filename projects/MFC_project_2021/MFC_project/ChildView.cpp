
// ChildView.cpp: CChildView 클래스의 구현
//

#include "pch.h"
#include "framework.h"
#include "MFC_project.h"
#include "ChildView.h"
#include <mmsystem.h>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CChildView

CChildView::CChildView()
{
	instruments = 0;//악기종류
	page = 0;//악보페이지
	current_page = 0;//현재 페이지
	a = 0;
	p_gap = 0;
	note_name.SetSize(8);//계명 array만들기
	note_name[0] = "도";
	note_name[1] = "레";
	note_name[2] = "미";
	note_name[3] = "파";
	note_name[4] = "솔";
	note_name[5] = "라";
	note_name[6] = "시";
	note_name[7] = "도";
}

CChildView::~CChildView()
{
}


BEGIN_MESSAGE_MAP(CChildView, CWnd)
	ON_WM_PAINT()
	ON_WM_LBUTTONDOWN()
	ON_COMMAND(ID_32772, &CChildView::piano_sound)
	ON_COMMAND(ID_32773, &CChildView::xylophone_sound)
END_MESSAGE_MAP()



// CChildView 메시지 처리기

BOOL CChildView::PreCreateWindow(CREATESTRUCT& cs) 
{
	if (!CWnd::PreCreateWindow(cs))
		return FALSE;

	cs.dwExStyle |= WS_EX_CLIENTEDGE;
	cs.style &= ~WS_BORDER;
	cs.lpszClass = AfxRegisterWndClass(CS_HREDRAW|CS_VREDRAW|CS_DBLCLKS, 
		::LoadCursor(nullptr, IDC_ARROW), reinterpret_cast<HBRUSH>(COLOR_WINDOW+1), nullptr);

	return TRUE;
}

void CChildView::OnPaint() 
{
	CPaintDC dc(this);
	GetClientRect(&clientview);    //화면 크기 얻어옴
	first_setting(&dc, clientview); //기본화면 그리기
	//draw_notes(2);//음표그리기( 임의로 지정한거 말고 다시 위치 지정해야함)
	POSITION pos;
	if (current_page == 0) {//현재 페이지가 0이면 0부터 시작
		pos = note_positions.GetHeadPosition();
	}
	else {//현재페이지가 0이 아니면 리스트 35*현재페이지부터 시작
		pos = note_positions.FindIndex(35*current_page);
	}
	
	CBrush brush(RGB(0, 0, 0)), * pbrush;
	pbrush = dc.SelectObject(&brush);
	while (pos != note_positions.FindIndex(35 * (current_page + 1))) {//시작 위치에서 35개만 그려지게
		CRect a = note_positions.GetNext(pos);
		dc.Ellipse(a);
	}
	
}

void CChildView::first_setting(CDC* dc, CRect clientview) {
	CBitmap bitmap;//높은 음 자리표 그리기
	bitmap.LoadBitmapW(IDB_BITMAP1);
	BITMAP bmpinfo;
	bitmap.GetBitmap(&bmpinfo);

	CDC dc_mem;
	dc_mem.CreateCompatibleDC(dc);
	dc_mem.SelectObject(&bitmap);
	dc->StretchBlt(50, clientview.Height() / 5 - 5 * gap, 70, 130, &dc_mem, 0, 0, bmpinfo.bmWidth, bmpinfo.bmHeight, SRCCOPY);

	for (int i = 0; i < 5; i++) {//오선지 그리기
		dc->MoveTo(50, clientview.Height() / 5 - gap * i);
		dc->LineTo(clientview.Width() - 50, clientview.Height() / 5 - gap * i);
	}
	p_gap = clientview.Width() / 3 - clientview.Width() / 2.5;

	for (int i = 0; i <= 7; i++) {//피아노 건반 그리기
		dc->Rectangle(clientview.Width() / 2 - (i - 3) * p_gap, clientview.Height() / 3, clientview.Width() / 2 - (i - 4) * p_gap, clientview.Height());
	}

	for (int i = 0; i <= 7; i++) {//피아노 건반 위치
		position_list.AddTail(CRect(clientview.Width() / 2 + (i - 3) * p_gap, clientview.Height() / 3, clientview.Width() / 2 + (i - 4) * p_gap, clientview.Height()));
	}

	for (int i = -4; i <= 3; i++) {//피아노 계명 그리기
		dc->TextOutW(clientview.Width() / 2 - i * p_gap - p_gap / 3, 4 * (clientview.Height() / 5), note_name[i + 4]);
	}
	removeAll_button = CRect(clientview.Width() * 0.845, clientview.Height() * 0.5, clientview.Width() * 0.96, clientview.Height() * 0.6);
	
	dc->Rectangle(removeAll_button);//사각형 그리기
	dc->TextOutW(clientview.Width() * 0.85, clientview.Height() * 0.53, _T("모든 음표 지우기"));//사각형안에 글씨

	remove_button = CRect(clientview.Width() * 0.845, clientview.Height() * 0.65, clientview.Width() * 0.96, clientview.Height() * 0.75);
	dc->Rectangle(remove_button);//사각형 그리기
	dc->TextOutW(clientview.Width() * 0.85, clientview.Height() * 0.68, _T("음표 하나 지우기"));//사각형안에 글씨

	next = CRect(clientview.Width() - 40, clientview.Height() / 5 - 50, clientview.Width() - 10, clientview.Height() / 5 - 20);
	dc->Rectangle(next);//사각형 그리기
	dc->TextOutW(clientview.Width() - 35, clientview.Height()/5 - 45, _T("▶"));//사각형안에 글씨

	prev = CRect(10, clientview.Height() / 5 - 50, 40, clientview.Height() / 5 - 20);
	dc->Rectangle(prev);//사각형 그리기
	dc->TextOutW(13, clientview.Height() / 5 - 45, _T("◀"));//사각형안에 글씨
}

void CChildView::draw_notes(int i) {//위치받는 리스트 필요
	CClientDC dc(this);
	CBrush brush(RGB(0, 0, 0)),* pbrush;
	pbrush = dc.SelectObject(&brush);

	if ((clientview.Width() - 50) < 150 + (a * 35)) { //오선줄을 넘었을때
		AfxMessageBox(_T("넘었습니다"));
		page++;//페이지 추가
		InvalidateRect(CRect(50, clientview.Height() / 10, clientview.Width() - 50, clientview.Height() / 4), true); //싹다 지우고 onpaint부르기
		a = 0; //음표 위치 처음부터 다시 시작
		current_page = page;
	}
	dc.Ellipse(150 + (a * 35), clientview.Height() / 5 - (i - 3)*(gap / 2), 150 + note + (a * 35), clientview.Height() / 5 - (i - 1)*(gap / 2)); 
	//35간격으로 음표 그리기 150은 width/10
	note_positions.AddTail(CRect(150 + (a * 35), clientview.Height() / 5 - (i - 3) * (gap / 2), 150 + note + (a * 35), clientview.Height() / 5 - (i - 1) * (gap / 2)));
	//그려진 음표 리스트에 저장
	a++;//음표 위치 옮김
	dc.SelectObject(pbrush);
	brush.DeleteObject();
}

void CChildView::OnLButtonDown(UINT nFlags, CPoint point)//피아노 눌렀을때 음이 무엇인지 출력, 음표그림
{
	CClientDC dc(this);
	POSITION pos = position_list.GetHeadPosition();
	int i = 7;
	while (pos != NULL) {
		if (position_list.GetNext(pos).PtInRect(point)) {
			draw_notes(i);
			current_point = point;
			make_sound(i, instruments);//음 출력함수
		}
		i--;
		if (i < 0)//이거 없으면 i가 0이하가 되서 오류 생김
			break;
	}

	if (removeAll_button.PtInRect(point)) {//눌렀을때 전체 음표 지우기
		note_positions.RemoveAll();
		InvalidateRect(CRect(50, clientview.Height() / 10, clientview.Width() - 50, clientview.Height() / 4), true);
		a = 0;//다 지웠으므로 맨 처음 위치로 조정
	}

	if (remove_button.PtInRect(point)) {//눌렀을때 음표 하나 지우기
		note_positions.RemoveTail();
		InvalidateRect(CRect(50, clientview.Height() / 10, clientview.Width() - 50, clientview.Height() / 4), true);
		a -= 1;;//하나 지웠으므로 그 전 위치로 조정
	}

	if (next.PtInRect(point)) {//누르면 다음 화면으로 넘어가기
		if (current_page == page) {
			AfxMessageBox(_T("마지막 페이지입니다."));
		}
		else {
			current_page++;//한페이지에 35개 음표 들어감
			InvalidateRect(CRect(50, clientview.Height() / 10, clientview.Width() - 50, clientview.Height() / 4), true);
		}
	}

	if (prev.PtInRect(point)) {//누르면 그전 화면으로 넘어가기
		if (current_page == 0) {
			AfxMessageBox(_T("맨 처음 페이지 입니다."));
		}
		else {
			current_page--;//한페이지에 35개 음표 들어감
			InvalidateRect(CRect(50, clientview.Height() / 10, clientview.Width() - 50, clientview.Height() / 4), true);
		}
	}
	CWnd::OnLButtonDown(nFlags, point);
}

void CChildView::make_sound(int i, int instruments) {//악기 음 출력
	if (instruments == 0) {//피아노
		switch (i) {
		case 0:
			szSoundPath = _T("C:/Users/USER/Desktop/MFC_project/soundfiles/do-c4.wav");
			break;
		case 1:
			szSoundPath = _T("C:/Users/USER/Desktop/MFC_project/soundfiles/re-d4.wav");
			break;
		case 2:
			szSoundPath = _T("C:/Users/USER/Desktop/MFC_project/soundfiles/mi-e4.wav");
			break;
		case 3:
			szSoundPath = _T("C:/Users/USER/Desktop/MFC_project/soundfiles/fa-f4.wav");
			break;
		case 4:
			szSoundPath = _T("C:/Users/USER/Desktop/MFC_project/soundfiles/sol-g4.wav");
			break;
		case 5:
			szSoundPath = _T("C:/Users/USER/Desktop/MFC_project/soundfiles/la-a4.wav");
			break;
		case 6:
			szSoundPath = _T("C:/Users/USER/Desktop/MFC_project/soundfiles/si-b4.wav");
			break;
		case 7:
			szSoundPath = _T("C:/Users/USER/Desktop/MFC_project/soundfiles/do-c5.wav");
			break;
		}
	}
	if (instruments == 1) {//실로폰
		switch (i) {
		case 0:
			szSoundPath = _T("C:/Users/USER/Desktop/MFC_project/soundfiles/xylophone-c1.wav");
			break;
		case 1:
			szSoundPath = _T("C:/Users/USER/Desktop/MFC_project/soundfiles/xylophone-d1.wav");
			break;
		case 2:
			szSoundPath = _T("C:/Users/USER/Desktop/MFC_project/soundfiles/xylophone-e1.wav");
			break;
		case 3:
			szSoundPath = _T("C:/Users/USER/Desktop/MFC_project/soundfiles/xylophone-f1.wav");
			break;
		case 4:
			szSoundPath = _T("C:/Users/USER/Desktop/MFC_project/soundfiles/xylophone-g1.wav");
			break;
		case 5:
			szSoundPath = _T("C:/Users/USER/Desktop/MFC_project/soundfiles/xylophone-a1.wav");
			break;
		case 6:
			szSoundPath = _T("C:/Users/USER/Desktop/MFC_project/soundfiles/xylophone-b1.wav");
			break;
		case 7:
			szSoundPath = _T("C:/Users/USER/Desktop/MFC_project/soundfiles/xylophone-c2.wav");
			break;
		}
	}
	PlaySound(szSoundPath, NULL, SND_ASYNC); // 1회 재생
}

void CChildView::piano_sound()//피아노 메뉴 선택했을때
{
	instruments = 0;
}


void CChildView::xylophone_sound()//실로폰 메뉴 선택했을때
{
	instruments = 1;
}
